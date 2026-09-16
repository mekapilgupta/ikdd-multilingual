#!/usr/bin/env python3
"""
translate_pipeline.py — Triple-Pass Multilingual Translation Pipeline
======================================================================
TARGET LANGUAGES: Odia (ory_Orya), Assamese (asm_Beng),
                  Sinhala (sin_Sinh), Maltese (mlt_Latn)

PASS 1A: Primary forward translation
  - Odia, Assamese  : IndicTrans2-en-indic-1B (best for Indic)
  - Sinhala, Maltese: NLLB-200-3.3B
PASS 1B: Secondary forward translation (independent second opinion)
  - All languages   : NLLB-200-3.3B  (if IndicTrans2 used in 1A)
                      NLLB-200-1.3B  (if NLLB-3.3B used in 1A)
PASS 2 : Back-translation + chrF++ + LaBSE (flag disagreements)
PASS 3 : vLLM Qwen3.5 judge on flagged items only

Fields translated per record: flores_passage, question, mc_answer1-4
Verification (back-translation + scoring) is done on flores_passage only.
"""
import os, gc, json, time, sys, logging, argparse
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("translate_pipeline")

# ─────────────────────────────────────────────────────────────────────────────
# Language Configuration
# ─────────────────────────────────────────────────────────────────────────────
LANG_CONFIG = {
    "ory_Orya": {
        "name": "Odia",
        "nllb_code": "ory_Orya",
        "use_indictrans_fwd": True,   # IndicTrans2 for Pass 1A
        "use_indictrans_back": True,  # IndicTrans2 for back-translation
    },
    "asm_Beng": {
        "name": "Assamese",
        "nllb_code": "asm_Beng",
        "use_indictrans_fwd": True,
        "use_indictrans_back": True,
    },
    "sin_Sinh": {
        "name": "Sinhala",
        "nllb_code": "sin_Sinh",
        "use_indictrans_fwd": False,  # NLLB-3.3B for Pass 1A
        "use_indictrans_back": False, # NLLB-1.3B for back-translation
    },
    "mlt_Latn": {
        "name": "Maltese",
        "nllb_code": "mlt_Latn",
        "use_indictrans_fwd": False,
        "use_indictrans_back": False,
    },
}

# Score thresholds for auto-accept vs. send to LLM judge
CHRF_AUTO_ACCEPT = 65.0   # chrF++ score (0-100 scale)
LABSE_AUTO_ACCEPT = 0.80  # cosine similarity (0-1 scale)
CHRF_GAP_FLAGGED = 15.0   # if |chrF_A - chrF_B| > this, flag for judge

# ─────────────────────────────────────────────────────────────────────────────
# NLLB Translation Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_nllb(model_path: str, device: str = "cuda"):
    import torch
    from transformers import AutoModelForSeq2SeqLM, NllbTokenizer, AutoTokenizer
    logger.info(f"Loading NLLB from {model_path}...")
    try:
        tokenizer = NllbTokenizer.from_pretrained(model_path, local_files_only=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_path, torch_dtype=torch.float16, local_files_only=True
    ).to(device).eval()
    logger.info(f"NLLB loaded on {device}.")
    return model, tokenizer

def translate_nllb(model, tokenizer, texts: List[str], src_lang: str, tgt_lang: str,
                   batch_size: int = 32, num_beams: int = 4, device: str = "cuda") -> List[str]:
    import torch
    tokenizer.src_lang = src_lang
    try:
        tgt_lang_id = tokenizer.convert_tokens_to_ids(tgt_lang)
    except Exception:
        tgt_lang_id = tokenizer.lang_code_to_id[tgt_lang]

    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True,
                           truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                forced_bos_token_id=tgt_lang_id,
                num_beams=num_beams,
                max_length=512,
                early_stopping=True
            )
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        results.extend(decoded)
        if (i // batch_size + 1) % 5 == 0:
            logger.info(f"  NLLB translated {min(i+batch_size,len(texts))}/{len(texts)}")
    return results

def unload_model(model):
    import torch
    del model
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("Model unloaded, GPU cache cleared.")

# ─────────────────────────────────────────────────────────────────────────────
# IndicTrans2 Helpers (Odia + Assamese)
# ─────────────────────────────────────────────────────────────────────────────

INDIC_TRANS_AVAILABLE = False
try:
    from IndicTransToolkit import IndicProcessor
    INDIC_TRANS_AVAILABLE = True
    logger.info("IndicTransToolkit available.")
except ImportError:
    logger.warning("IndicTransToolkit not found — Odia/Assamese will use NLLB-3.3B instead.")

def load_indictrans(model_path: str, device: str = "cuda"):
    if not INDIC_TRANS_AVAILABLE:
        return None, None, None
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    logger.info(f"Loading IndicTrans2 from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=torch.float16,
        local_files_only=True
    ).to(device).eval()
    ip = IndicProcessor(inference=True)
    logger.info("IndicTrans2 loaded.")
    return model, tokenizer, ip

def translate_indictrans(model, tokenizer, ip, texts: List[str],
                          src_lang: str, tgt_lang: str,
                          batch_size: int = 32, device: str = "cuda") -> List[str]:
    import torch
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        batch_in = ip.preprocess_batch(batch, src_lang=src_lang, tgt_lang=tgt_lang)
        inputs = tokenizer(batch_in, return_tensors="pt", padding=True,
                           truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, num_beams=4, max_length=512,
                                      min_length=1, early_stopping=True)
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        translated = ip.postprocess_batch(decoded, lang=tgt_lang)
        results.extend(translated)
        if (i // batch_size + 1) % 5 == 0:
            logger.info(f"  IndicTrans2 translated {min(i+batch_size,len(texts))}/{len(texts)}")
    return results

# ─────────────────────────────────────────────────────────────────────────────
# LaBSE Semantic Similarity
# ─────────────────────────────────────────────────────────────────────────────

def compute_labse_scores(labse_model, texts_en: List[str], texts_tgt: List[str],
                          batch_size: int = 64) -> List[float]:
    import torch
    emb_en = labse_model.encode(texts_en, batch_size=batch_size,
                                  convert_to_tensor=True, normalize_embeddings=True)
    emb_tgt = labse_model.encode(texts_tgt, batch_size=batch_size,
                                   convert_to_tensor=True, normalize_embeddings=True)
    scores = (emb_en * emb_tgt).sum(dim=1).cpu().tolist()
    return scores

# ─────────────────────────────────────────────────────────────────────────────
# chrF++ Scoring (CPU, microseconds per sentence)
# ─────────────────────────────────────────────────────────────────────────────

def _simple_char_chrf(hyp: str, ref: str, max_n: int = 6, beta: float = 2.0) -> float:
    """Pure Python fallback for chrF calculation (0-100 scale)."""
    if not hyp or not ref:
        return 0.0
    precisions, recalls = [], []
    for n in range(1, max_n + 1):
        hyp_ngrams = [hyp[i:i+n] for i in range(len(hyp) - n + 1)]
        ref_ngrams = [ref[i:i+n] for i in range(len(ref) - n + 1)]
        if not hyp_ngrams or not ref_ngrams:
            continue
        from collections import Counter
        hyp_counts = Counter(hyp_ngrams)
        ref_counts = Counter(ref_ngrams)
        overlap = sum((hyp_counts & ref_counts).values())
        p = overlap / len(hyp_ngrams) if hyp_ngrams else 0
        r = overlap / len(ref_ngrams) if ref_ngrams else 0
        precisions.append(p)
        recalls.append(r)
    avg_p = sum(precisions) / len(precisions) if precisions else 0
    avg_r = sum(recalls) / len(recalls) if recalls else 0
    if avg_p + avg_r == 0:
        return 0.0
    f_beta = (1 + beta**2) * (avg_p * avg_r) / ((beta**2 * avg_p) + avg_r)
    return f_beta * 100.0

def compute_chrf_scores(hypotheses: List[str], references: List[str]) -> List[float]:
    try:
        import sacrebleu
        use_sacrebleu = True
    except ImportError:
        logger.warning('sacrebleu module not installed — using pure Python chrF fallback.')
        use_sacrebleu = False

    scores = []
    for hyp, ref in zip(hypotheses, references):
        try:
            if use_sacrebleu:
                score = sacrebleu.sentence_chrf(hyp, [ref]).score
            else:
                score = _simple_char_chrf(hyp, ref)
        except Exception as e:
            logger.warning(f'chrF error: {e}')
            score = _simple_char_chrf(hyp, ref)
        scores.append(score)
    return scores

# ─────────────────────────────────────────────────────────────────────────────
# vLLM LLM Judge (Pass 3) — uses Qwen3.5 with thinking
# ─────────────────────────────────────────────────────────────────────────────

def build_judge_prompt(tokenizer, original_en: str, trans_a: str, trans_b: str,
                        back_a: str, back_b: str, chrf_a: float, chrf_b: float,
                        labse_a: float, labse_b: float, tgt_lang_name: str) -> str:
    system_msg = (
        "You are an expert multilingual translation quality evaluator. "
        "Your task is to select the better translation based ONLY on faithfulness "
        "to the original English text — meaning, semantic accuracy and completeness, "
        "not just fluency."
    )
    user_msg = (
        f"TARGET LANGUAGE: {tgt_lang_name}\n\n"
        f"ORIGINAL ENGLISH:\n{original_en}\n\n"
        f"TRANSLATION A:\n{trans_a}\n"
        f"  Back-translated to English: {back_a}\n"
        f"  chrF++ back-translation score: {chrf_a:.1f}/100\n"
        f"  LaBSE semantic similarity: {labse_a:.3f}\n\n"
        f"TRANSLATION B:\n{trans_b}\n"
        f"  Back-translated to English: {back_b}\n"
        f"  chrF++ back-translation score: {chrf_b:.1f}/100\n"
        f"  LaBSE semantic similarity: {labse_b:.3f}\n\n"
        "EVALUATION TASK:\n"
        "1. Compare what meaning was LOST or CHANGED in each translation by comparing "
        "   the back-translations to the original.\n"
        "2. Select the translation that better preserves the original meaning.\n"
        "3. If neither is acceptable (both chrF < 40), flag as REJECT.\n\n"
        "Output ONLY valid JSON:\n"
        '{"winner": "A" | "B" | "REJECT", '
        '"confidence": "high" | "medium" | "low", '
        '"errors_in_loser": "brief description of what was lost/changed", '
        '"accepted_translation": "the full text of the winning translation or empty if REJECT"}'
    )
    messages = [{"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        try:
            return tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                                  tokenize=False, enable_thinking=True)
        except TypeError:
            try:
                return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            except: pass
        except: pass
    return (f"<|im_start|>system\n{system_msg}<|im_end|>\n"
            f"<|im_start|>user\n{user_msg}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n")

def parse_judge_response(text: str) -> Optional[dict]:
    """Extract JSON from LLM judge response (may contain thinking chain)."""
    try:
        cleaned = text.strip()
        if "<think>" in cleaned:
            think_end = cleaned.rfind("</think>")
            if think_end != -1:
                cleaned = cleaned[think_end + len("</think>"):].strip()
        start = cleaned.find("{")
        if start != -1:
            cleaned = cleaned[start:]
        data = json.loads(cleaned)
        if data.get("winner") not in {"A", "B", "REJECT"}:
            return None
        return data
    except Exception:
        return None

# ─────────────────────────────────────────────────────────────────────────────
# Intermediate Results Cache (resume logic)
# ─────────────────────────────────────────────────────────────────────────────

def load_cache(cache_path: str) -> dict:
    """Load cached translation results keyed by (item_id, lang_code, pass)."""
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entry = json.loads(line)
                        key = (entry["item_id"], entry["lang_code"], entry["pass"])
                        cache[key] = entry
                    except: pass
        logger.info(f"Loaded {len(cache)} cached entries from {cache_path}")
    return cache

def save_cache_entry(cache_path: str, entry: dict):
    with open(cache_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        f.flush()

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Triple-pass multilingual translation pipeline.")
    parser.add_argument("--sampled-file", default="sampled_eng_Latn.jsonl")
    parser.add_argument("--output-dir", default="translated")
    parser.add_argument("--cache-file", default="translate_cache.jsonl")
    parser.add_argument("--nllb-large-path", default="facebook/nllb-200-3.3B")
    parser.add_argument("--nllb-small-path", default="facebook/nllb-200-1.3B")
    parser.add_argument("--indictrans-fwd-path", default="ai4bharat/indictrans2-en-indic-1B")
    parser.add_argument("--indictrans-back-path", default="ai4bharat/indictrans2-indic-en-1B")
    parser.add_argument("--labse-path", default="sentence-transformers/LaBSE")
    parser.add_argument("--judge-model-path", default="Qwen/Qwen2.5-32B-Instruct")
    parser.add_argument("--languages", nargs="+",
                        default=["ory_Orya", "asm_Beng", "sin_Sinh", "mlt_Latn"])
    parser.add_argument("--nmt-batch-size", type=int, default=32)
    parser.add_argument("--judge-batch-size", type=int, default=16)
    parser.add_argument("--judge-max-tokens", type=int, default=2048)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if not os.path.exists(args.sampled_file):
        logger.error(f"Sampled file not found: {args.sampled_file}. Run sample_dataset.py first.")
        sys.exit(1)

    # Load sampled English data
    items = []
    with open(args.sampled_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    logger.info(f"Loaded {len(items)} sampled items.")

    # All text fields to translate per item
    TEXT_FIELDS = ["flores_passage", "question", "mc_answer1", "mc_answer2",
                   "mc_answer3", "mc_answer4"]

    # Load cache for resume
    cache = load_cache(args.cache_file)

    # =========================================================================
    # PHASE A: NMT TRANSLATION (Pass 1A + 1B for all languages)
    # =========================================================================

    # ── Determine which languages need IndicTrans2 ──
    langs_needing_indictrans = [l for l in args.languages
                                 if LANG_CONFIG[l]["use_indictrans_fwd"] and INDIC_TRANS_AVAILABLE]
    langs_using_nllb_fwd = [l for l in args.languages
                              if not (LANG_CONFIG[l]["use_indictrans_fwd"] and INDIC_TRANS_AVAILABLE)]

    # ── PASS 1A with NLLB-3.3B for non-Indic languages ──
    if langs_using_nllb_fwd:
        logger.info("=" * 70)
        logger.info(f"PASS 1A: NLLB-3.3B for {langs_using_nllb_fwd}")
        logger.info("=" * 70)
        nllb_large, nllb_large_tok = load_nllb(args.nllb_large_path, args.device)

        for lang_code in langs_using_nllb_fwd:
            cfg = LANG_CONFIG[lang_code]
            needed = [it for it in items
                      if (it["item_id"], lang_code, "1A") not in cache]
            if not needed:
                logger.info(f"[{lang_code}] Pass 1A already cached.")
                continue
            logger.info(f"[{lang_code}] Pass 1A: translating {len(needed)} items...")
            for field in TEXT_FIELDS:
                texts = [it.get(field, "") for it in needed]
                translations = translate_nllb(
                    nllb_large, nllb_large_tok, texts,
                    src_lang="eng_Latn", tgt_lang=lang_code,
                    batch_size=args.nmt_batch_size, device=args.device
                )
                for it, tr in zip(needed, translations):
                    key = (it["item_id"], lang_code, "1A")
                    if key not in cache:
                        cache[key] = {"item_id": it["item_id"], "lang_code": lang_code,
                                      "pass": "1A", "translations": {}}
                    cache[key]["translations"][field] = tr
            # Save 1A entries
            for it in needed:
                key = (it["item_id"], lang_code, "1A")
                if key in cache:
                    save_cache_entry(args.cache_file, cache[key])
            logger.info(f"[{lang_code}] Pass 1A done.")

        # PASS 1B for NLLB-forward languages: use NLLB-large with more sampling
        # (same model, different decoding: higher temperature sampling)
        for lang_code in langs_using_nllb_fwd:
            needed = [it for it in items
                      if (it["item_id"], lang_code, "1B") not in cache]
            if not needed:
                logger.info(f"[{lang_code}] Pass 1B already cached.")
                continue
            logger.info(f"[{lang_code}] Pass 1B: NLLB-3.3B with sampling (n=1, temp=1.0)...")
            for field in TEXT_FIELDS:
                texts = [it.get(field, "") for it in needed]
                # Use beam=1 + sampling for diversity from same model
                import torch
                nllb_large_tok.src_lang = "eng_Latn"
                try:
                    tgt_id = nllb_large_tok.convert_tokens_to_ids(lang_code)
                except:
                    tgt_id = nllb_large_tok.lang_code_to_id[lang_code]
                results = []
                for i in range(0, len(texts), args.nmt_batch_size):
                    batch = texts[i:i+args.nmt_batch_size]
                    enc = nllb_large_tok(batch, return_tensors="pt", padding=True,
                                         truncation=True, max_length=512).to(args.device)
                    with torch.no_grad():
                        out = nllb_large.generate(
                            **enc, forced_bos_token_id=tgt_id,
                            do_sample=True, temperature=0.8, top_k=50, max_length=512
                        )
                    results.extend(nllb_large_tok.batch_decode(out, skip_special_tokens=True))

                for it, tr in zip(needed, results):
                    key = (it["item_id"], lang_code, "1B")
                    if key not in cache:
                        cache[key] = {"item_id": it["item_id"], "lang_code": lang_code,
                                      "pass": "1B", "translations": {}}
                    cache[key]["translations"][field] = tr
            for it in needed:
                key = (it["item_id"], lang_code, "1B")
                if key in cache:
                    save_cache_entry(args.cache_file, cache[key])
            logger.info(f"[{lang_code}] Pass 1B done.")

        unload_model(nllb_large)

    # ── PASS 1A with IndicTrans2 for Odia + Assamese ──
    if langs_needing_indictrans:
        logger.info("=" * 70)
        logger.info(f"PASS 1A: IndicTrans2 for {langs_needing_indictrans}")
        logger.info("=" * 70)
        it2_fwd, it2_fwd_tok, it2_ip = load_indictrans(args.indictrans_fwd_path, args.device)
        if it2_fwd is not None:
            for lang_code in langs_needing_indictrans:
                needed = [it for it in items
                          if (it["item_id"], lang_code, "1A") not in cache]
                if not needed:
                    logger.info(f"[{lang_code}] Pass 1A already cached.")
                    continue
                logger.info(f"[{lang_code}] Pass 1A: IndicTrans2 for {len(needed)} items...")
                for field in TEXT_FIELDS:
                    texts = [it.get(field, "") for it in needed]
                    translations = translate_indictrans(
                        it2_fwd, it2_fwd_tok, it2_ip, texts,
                        src_lang="eng_Latn", tgt_lang=lang_code,
                        batch_size=args.nmt_batch_size, device=args.device
                    )
                    for item, tr in zip(needed, translations):
                        key = (item["item_id"], lang_code, "1A")
                        if key not in cache:
                            cache[key] = {"item_id": item["item_id"], "lang_code": lang_code,
                                          "pass": "1A", "model": "indictrans2", "translations": {}}
                        cache[key]["translations"][field] = tr
                for item in needed:
                    key = (item["item_id"], lang_code, "1A")
                    if key in cache:
                        save_cache_entry(args.cache_file, cache[key])
                logger.info(f"[{lang_code}] Pass 1A (IndicTrans2) done.")
            unload_model(it2_fwd)

        # PASS 1B with NLLB-3.3B for Indic languages (independent second opinion)
        logger.info(f"PASS 1B: NLLB-3.3B for Indic languages {langs_needing_indictrans}")
        nllb_large, nllb_large_tok = load_nllb(args.nllb_large_path, args.device)
        for lang_code in langs_needing_indictrans:
            needed = [it for it in items
                      if (it["item_id"], lang_code, "1B") not in cache]
            if not needed:
                logger.info(f"[{lang_code}] Pass 1B already cached.")
                continue
            logger.info(f"[{lang_code}] Pass 1B: NLLB-3.3B for {len(needed)} items...")
            for field in TEXT_FIELDS:
                texts = [it.get(field, "") for it in needed]
                translations = translate_nllb(
                    nllb_large, nllb_large_tok, texts,
                    src_lang="eng_Latn", tgt_lang=lang_code,
                    batch_size=args.nmt_batch_size, device=args.device
                )
                for item, tr in zip(needed, translations):
                    key = (item["item_id"], lang_code, "1B")
                    if key not in cache:
                        cache[key] = {"item_id": item["item_id"], "lang_code": lang_code,
                                      "pass": "1B", "model": "nllb-3.3B", "translations": {}}
                    cache[key]["translations"][field] = tr
            for item in needed:
                key = (item["item_id"], lang_code, "1B")
                if key in cache:
                    save_cache_entry(args.cache_file, cache[key])
            logger.info(f"[{lang_code}] Pass 1B (NLLB-3.3B) done.")
        unload_model(nllb_large)

    # ── Back-translation (passages only) ──
    logger.info("=" * 70)
    logger.info("PASS 2A: BACK-TRANSLATION of passages")
    logger.info("=" * 70)

    # Determine which back-model to use per language
    langs_nllb_back = [l for l in args.languages if not LANG_CONFIG[l]["use_indictrans_back"]
                        or not INDIC_TRANS_AVAILABLE]
    langs_indic_back = [l for l in args.languages if LANG_CONFIG[l]["use_indictrans_back"]
                         and INDIC_TRANS_AVAILABLE]

    if langs_nllb_back:
        nllb_small, nllb_small_tok = load_nllb(args.nllb_small_path, args.device)
        for lang_code in langs_nllb_back:
            for pass_tag in ["1A", "1B"]:
                needed = [it for it in items
                          if (it["item_id"], lang_code, f"back_{pass_tag}") not in cache
                          and (it["item_id"], lang_code, pass_tag) in cache]
                if not needed:
                    logger.info(f"[{lang_code}] Back-{pass_tag} already cached.")
                    continue
                logger.info(f"[{lang_code}] Back-{pass_tag}: {len(needed)} passages...")
                texts = [cache[(it["item_id"], lang_code, pass_tag)]["translations"]["flores_passage"]
                         for it in needed]
                back_texts = translate_nllb(
                    nllb_small, nllb_small_tok, texts,
                    src_lang=lang_code, tgt_lang="eng_Latn",
                    batch_size=args.nmt_batch_size, device=args.device
                )
                for item, bt in zip(needed, back_texts):
                    entry = {"item_id": item["item_id"], "lang_code": lang_code,
                              "pass": f"back_{pass_tag}", "back_passage": bt}
                    cache[(item["item_id"], lang_code, f"back_{pass_tag}")] = entry
                    save_cache_entry(args.cache_file, entry)
                logger.info(f"[{lang_code}] Back-{pass_tag} done.")
        unload_model(nllb_small)

    if langs_indic_back:
        it2_back, it2_back_tok, it2_back_ip = load_indictrans(args.indictrans_back_path, args.device)
        if it2_back is not None:
            for lang_code in langs_indic_back:
                for pass_tag in ["1A", "1B"]:
                    # For 1A (IndicTrans2 fwd), use IndicTrans2 back
                    # For 1B (NLLB-3.3B fwd), back-translated with NLLB already above
                    if pass_tag == "1B":
                        continue  # NLLB-fwd output -> back with NLLB
                    needed = [it for it in items
                              if (it["item_id"], lang_code, f"back_{pass_tag}") not in cache
                              and (it["item_id"], lang_code, pass_tag) in cache]
                    if not needed:
                        continue
                    logger.info(f"[{lang_code}] Back-{pass_tag} IndicTrans2: {len(needed)} passages...")
                    texts = [cache[(it["item_id"], lang_code, pass_tag)]["translations"]["flores_passage"]
                             for it in needed]
                    back_texts = translate_indictrans(
                        it2_back, it2_back_tok, it2_back_ip, texts,
                        src_lang=lang_code, tgt_lang="eng_Latn",
                        batch_size=args.nmt_batch_size, device=args.device
                    )
                    for item, bt in zip(needed, back_texts):
                        entry = {"item_id": item["item_id"], "lang_code": lang_code,
                                  "pass": f"back_{pass_tag}", "back_passage": bt}
                        cache[(item["item_id"], lang_code, f"back_{pass_tag}")] = entry
                        save_cache_entry(args.cache_file, entry)
            unload_model(it2_back)

    # =========================================================================
    # PHASE B: SCORING (chrF + LaBSE)
    # =========================================================================
    logger.info("=" * 70)
    logger.info("PASS 2B: chrF++ + LaBSE Scoring")
    logger.info("=" * 70)

    # chrF (CPU, instant)
    scores = {}  # (item_id, lang_code) -> {chrf_a, chrf_b, labse_a, labse_b}
    for item in items:
        for lang_code in args.languages:
            key = (item["item_id"], lang_code)
            orig = item.get("flores_passage", "")
            back_a = cache.get((item["item_id"], lang_code, "back_1A"), {}).get("back_passage", "")
            back_b = cache.get((item["item_id"], lang_code, "back_1B"), {}).get("back_passage", "")
            if not back_a and not back_b:
                continue
            chrf_a = compute_chrf_scores([back_a], [orig])[0] if back_a else 0.0
            chrf_b = compute_chrf_scores([back_b], [orig])[0] if back_b else 0.0
            scores[key] = {"chrf_a": chrf_a, "chrf_b": chrf_b}

    logger.info(f"chrF computed for {len(scores)} (item, lang) pairs.")

    # LaBSE
    from sentence_transformers import SentenceTransformer
    logger.info(f"Loading LaBSE from {args.labse_path}...")
    labse = SentenceTransformer(args.labse_path, device=args.device)

    for lang_code in args.languages:
        lang_items = [it for it in items
                      if (it["item_id"], lang_code) in scores]
        if not lang_items:
            continue
        logger.info(f"[{lang_code}] Computing LaBSE for {len(lang_items)} items...")
        orig_passages = [it["flores_passage"] for it in lang_items]
        trans_a = [cache.get((it["item_id"], lang_code, "1A"), {}).get("translations", {}).get("flores_passage", "")
                   for it in lang_items]
        trans_b = [cache.get((it["item_id"], lang_code, "1B"), {}).get("translations", {}).get("flores_passage", "")
                   for it in lang_items]
        labse_a = compute_labse_scores(labse, orig_passages, trans_a)
        labse_b = compute_labse_scores(labse, orig_passages, trans_b)
        for it, la, lb in zip(lang_items, labse_a, labse_b):
            scores[(it["item_id"], lang_code)]["labse_a"] = la
            scores[(it["item_id"], lang_code)]["labse_b"] = lb

    del labse
    gc.collect()
    import torch; torch.cuda.empty_cache()
    logger.info("LaBSE scoring complete, model unloaded.")

    # =========================================================================
    # FLAGGING: Decide auto-accept vs. send to LLM judge
    # =========================================================================
    logger.info("=" * 70)
    logger.info("FLAGGING: Auto-accept vs. LLM judge")
    logger.info("=" * 70)

    flagged = []    # items needing LLM judge
    decisions = {}  # (item_id, lang_code) -> {"winner": "A"/"B", "auto": True/False}

    for item in items:
        for lang_code in args.languages:
            key = (item["item_id"], lang_code)
            if key not in scores:
                continue
            sc = scores[key]
            chrf_a = sc.get("chrf_a", 0)
            chrf_b = sc.get("chrf_b", 0)
            labse_a = sc.get("labse_a", 0)
            labse_b = sc.get("labse_b", 0)
            chrf_gap = abs(chrf_a - chrf_b)

            # Auto-accept: both high or clearly one is better
            if chrf_a >= CHRF_AUTO_ACCEPT and labse_a >= LABSE_AUTO_ACCEPT \
               and chrf_b >= CHRF_AUTO_ACCEPT and labse_b >= LABSE_AUTO_ACCEPT \
               and chrf_gap <= CHRF_GAP_FLAGGED:
                # Both good and agree → pick A (primary model)
                decisions[key] = {"winner": "A", "auto": True}
            elif chrf_a >= CHRF_AUTO_ACCEPT and labse_a >= LABSE_AUTO_ACCEPT \
                 and (chrf_b < CHRF_AUTO_ACCEPT or labse_b < LABSE_AUTO_ACCEPT):
                decisions[key] = {"winner": "A", "auto": True}
            elif chrf_b >= CHRF_AUTO_ACCEPT and labse_b >= LABSE_AUTO_ACCEPT \
                 and (chrf_a < CHRF_AUTO_ACCEPT or labse_a < LABSE_AUTO_ACCEPT):
                decisions[key] = {"winner": "B", "auto": True}
            else:
                # Disagreement or both low → LLM judge
                flagged.append((item, lang_code))

    auto_count = sum(1 for d in decisions.values() if d["auto"])
    logger.info(f"Auto-accepted: {auto_count} | Flagged for LLM judge: {len(flagged)}")
    
    # Save intermediate scores and decisions
    scores_export = {f"{k[0]}|{k[1]}": v for k, v in scores.items()}
    with open("ikdd/translate_scores.json", "w", encoding="utf-8") as f:
        json.dump(scores_export, f, ensure_ascii=False, indent=2)
    decisions_export = {f"{k[0]}|{k[1]}": v for k, v in decisions.items()}
    with open("ikdd/translate_decisions.json", "w", encoding="utf-8") as f:
        json.dump(decisions_export, f, ensure_ascii=False, indent=2)
    logger.info("Saved intermediate scores and decisions to ikdd/")


    logger.info("Phase C (LLM Judge) is now handled by ikdd/translate_judge.py in a separate process.")
    logger.info(f"  {len(flagged)} items flagged — run translate_judge.py to adjudicate them.")

    # =========================================================================
    # PHASE D: FINALIZE — Write per-language output JSONL
    # =========================================================================
    logger.info("=" * 70)
    logger.info("FINALIZING: Writing per-language output files")
    logger.info("=" * 70)

    stats = defaultdict(lambda: {"accepted_A": 0, "accepted_B": 0, "judge_corrected": 0, "rejected": 0})

    for lang_code in args.languages:
        out_path = os.path.join(args.output_dir, f"translated_{lang_code}.jsonl")
        written = 0
        with open(out_path, "w", encoding="utf-8") as fout:
            for item in items:
                key = (item["item_id"], lang_code)
                if key not in decisions:
                    continue
                dec = decisions[key]
                winner = dec["winner"]
                pass_tag = "1A" if winner == "A" else "1B"
                trans_entry = cache.get((item["item_id"], lang_code, pass_tag), {})
                trans = trans_entry.get("translations", {})
                sc = scores.get(key, {})

                # If judge provided a corrected translation, use it for passage
                judge_accepted = dec.get("judge_accepted", "")
                final_passage = (judge_accepted if judge_accepted else
                                  trans.get("flores_passage", ""))

                record = {
                    "item_id": item["item_id"],
                    "lang_code": lang_code,
                    # English originals
                    "flores_passage_en": item.get("flores_passage", ""),
                    "question_en": item.get("question", ""),
                    "mc_answer1_en": item.get("mc_answer1", ""),
                    "mc_answer2_en": item.get("mc_answer2", ""),
                    "mc_answer3_en": item.get("mc_answer3", ""),
                    "mc_answer4_en": item.get("mc_answer4", ""),
                    # Translated
                    "flores_passage": final_passage,
                    "question": trans.get("question", ""),
                    "mc_answer1": trans.get("mc_answer1", ""),
                    "mc_answer2": trans.get("mc_answer2", ""),
                    "mc_answer3": trans.get("mc_answer3", ""),
                    "mc_answer4": trans.get("mc_answer4", ""),
                    # Labels (unchanged)
                    "correct_answer_num": item.get("correct_answer_num", ""),
                    "reasoning_level": item.get("reasoning_level", ""),
                    "error_category": item.get("error_category", ""),
                    "link": item.get("link", ""),
                    "ds": item.get("ds", ""),
                    # Quality metadata
                    "translation_model_winner": f"1A ({trans_entry.get('model','nllb-3.3B')})"
                                                 if winner == "A" else "1B (nllb-3.3B)",
                    "chrf_a": round(sc.get("chrf_a", 0), 2),
                    "chrf_b": round(sc.get("chrf_b", 0), 2),
                    "labse_a": round(sc.get("labse_a", 0), 4),
                    "labse_b": round(sc.get("labse_b", 0), 4),
                    "auto_accepted": dec.get("auto", True),
                    "judge_confidence": dec.get("judge_confidence", ""),
                    "judge_errors_in_loser": dec.get("judge_errors", ""),
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1

                if winner == "A":
                    stats[lang_code]["accepted_A"] += 1
                else:
                    stats[lang_code]["accepted_B"] += 1
                if not dec.get("auto", True):
                    stats[lang_code]["judge_corrected"] += 1

        logger.info(f"[{lang_code}] Wrote {written} records to {out_path}")
        st = stats[lang_code]
        logger.info(f"  Model A wins: {st['accepted_A']} | Model B wins: {st['accepted_B']} | "
                    f"Judge-adjudicated: {st['judge_corrected']}")

    logger.info("=" * 70)
    logger.info("TRANSLATION PIPELINE COMPLETE")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("=" * 70)

if __name__ == "__main__":
    main()
