#!/usr/bin/env python3
"""
quality_correction.py — Frontier-LLM Translation Quality Correction Pipeline
==============================================================================
Stage 1: google/gemini-3.7-flash:batch  → retranslate all 3600 items
Stage 2: qwen/qwen3-7-flash             → targeted fix for Stage-1 failures
Stage 3: Re-score with chrF++ + LaBSE  → rollback if worse

Usage:
  python ikdd/quality_correction.py                    # full run
  python ikdd/quality_correction.py --pilot-n 10       # pilot 10 items per language
  python ikdd/quality_correction.py --rescore-only     # re-score from cache, no API
"""

import os, gc, sys, json, time, asyncio, argparse, logging, traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ── API config ────────────────────────────────────────────────────────────────
API_KEY      = os.environ.get("OPENROUTER_API_KEY", "")
BASE_URL     = "https://openrouter.ai/api/v1/chat/completions"
GEMINI_MODEL = "google/gemini-3.7-flash:batch"
QWEN_MODEL   = "qwen/qwen3.7-flash"

LANG_NAMES = {
    "ory_Orya": "Odia",
    "asm_Beng": "Assamese",
    "sin_Sinh": "Sinhala",
    "mlt_Latn": "Maltese",
}

FIELDS_TO_TRANSLATE = [
    "flores_passage", "question",
    "mc_answer1", "mc_answer2", "mc_answer3", "mc_answer4",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("ikdd/correction_pipeline.log", mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger("correction")


# ─────────────────────────────────────────────────────────────────────────────
# chrF++ (pure Python, no sacrebleu dependency)
# ─────────────────────────────────────────────────────────────────────────────
def _char_ngrams(text: str, n: int) -> dict:
    ng = {}
    for i in range(len(text) - n + 1):
        k = text[i:i+n]
        ng[k] = ng.get(k, 0) + 1
    return ng

def chrf_score(hyp: str, ref: str, max_n: int = 6, beta: float = 2.0) -> float:
    if not hyp or not ref:
        return 0.0
    precs, recs = [], []
    for n in range(1, max_n + 1):
        h = _char_ngrams(hyp, n); r = _char_ngrams(ref, n)
        if not h or not r:
            continue
        overlap = sum(min(h.get(k, 0), r.get(k, 0)) for k in set(h) & set(r))
        precs.append(overlap / sum(h.values()))
        recs.append(overlap / sum(r.values()))
    if not precs:
        return 0.0
    p = sum(precs) / len(precs); r_ = sum(recs) / len(recs)
    if p + r_ == 0:
        return 0.0
    return (1 + beta**2) * p * r_ / (beta**2 * p + r_) * 100.0


# ─────────────────────────────────────────────────────────────────────────────
# Cache helpers
# ─────────────────────────────────────────────────────────────────────────────
def load_cache(cache_file: str) -> dict:
    cache = {}
    if not os.path.exists(cache_file):
        return cache
    with open(cache_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                cache[(entry["item_id"], entry["lang_code"], entry["model"])] = entry
            except Exception:
                pass
    return cache

def save_cache_entry(cache_file: str, entry: dict):
    with open(cache_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Token usage logging
# ─────────────────────────────────────────────────────────────────────────────
def log_tokens(token_log: str, model: str, stage: str, n_items: int,
               prompt_tokens: int, completion_tokens: int, elapsed_s: float):
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model, "stage": stage,
        "n_items": n_items,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "elapsed_s": round(elapsed_s, 2),
        "tokens_per_item": round((prompt_tokens + completion_tokens) / max(n_items, 1), 1),
    }
    with open(token_log, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    logger.info(
        f"[Token] {model} | {stage} | {n_items} items | "
        f"prompt={prompt_tokens:,} comp={completion_tokens:,} "
        f"total={entry['total_tokens']:,} | {elapsed_s:.1f}s | "
        f"{entry['tokens_per_item']:.0f} tok/item"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Prompt builder
# ─────────────────────────────────────────────────────────────────────────────
def build_correction_prompt(item: dict, lang_code: str, model: str) -> str:
    lang = LANG_NAMES.get(lang_code, lang_code)
    no_think = "/no_think\n" if "qwen" in model.lower() else ""

    en_block = "\n".join(
        f'{f.replace("flores_passage", "passage").replace("mc_", "option_")}: '
        f'{item.get(f + "_en", item.get(f, ""))}'
        for f in FIELDS_TO_TRANSLATE
    )
    cur_block = "\n".join(
        f'{f.replace("flores_passage", "passage").replace("mc_", "option_")}: '
        f'{item.get(f, "")}'
        for f in FIELDS_TO_TRANSLATE
    )

    return (
        f"{no_think}"
        f"Retranslate into {lang}. Output ONLY a JSON object with these exact keys: "
        f"{FIELDS_TO_TRANSLATE}. No explanation. No markdown.\n\n"
        f"ENGLISH:\n{en_block}\n\n"
        f"CURRENT {lang.upper()} (may have errors):\n{cur_block}\n\n"
        f"CORRECTED {lang.upper()} JSON:"
    )


# ─────────────────────────────────────────────────────────────────────────────
# API call (async, single item)
# ─────────────────────────────────────────────────────────────────────────────
async def call_api(session, model: str, prompt: str,
                   max_tokens: int = 4096, temperature: float = 0.1,
                   retries: int = 4) -> tuple:
    import httpx
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://ikdd.datasets",
        "X-Title": "IKDD-Translation-Correction",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 4096,
        "temperature": temperature,
        "reasoning": {"max_tokens": 128},
    }
    for attempt in range(retries):
        try:
            r = await session.post(BASE_URL, json=payload, headers=headers, timeout=120.0)
            if r.status_code != 200:
                wait = 3 * (attempt + 1)
                logger.warning(f"[{model}] HTTP {r.status_code} (attempt {attempt+1}/{retries}): {r.text[:150]}... waiting {wait}s")
                await asyncio.sleep(wait)
                continue
            data = r.json()
            text = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            return text, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
        except Exception as e:
            if attempt < retries - 1:
                await asyncio.sleep(2 ** attempt)
            else:
                logger.error(f"[{model}] API failed after {retries} attempts: {e}")
    return None, 0, 0


# ─────────────────────────────────────────────────────────────────────────────
# Parse JSON from model output
# ─────────────────────────────────────────────────────────────────────────────
def parse_translation_response(text: str) -> Optional[dict]:
    import re
    if not text:
        return None
    text = re.sub(r"```(?:json)?\s*", "", text).replace("```", "").strip()
    # Try from last { to last }
    brace_positions = [i for i, c in enumerate(text) if c == "{"]
    for start in reversed(brace_positions):
        end = text.rfind("}", start)
        if end == -1:
            continue
        try:
            data = json.loads(text[start:end+1])
            if isinstance(data, dict) and any(k in data for k in FIELDS_TO_TRANSLATE):
                return data
        except Exception:
            pass
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Correct batch (async)
# ─────────────────────────────────────────────────────────────────────────────
async def correct_batch(items_langs, model, cache, cache_file, token_log,
                        stage, concurrency=20):
    import httpx
    results = {}
    to_process = []

    for item, lang in items_langs:
        key = (item["item_id"], lang, model)
        if key in cache:
            results[(item["item_id"], lang)] = cache[key].get("corrected_fields", {})
        else:
            to_process.append((item, lang))

    logger.info(f"[{stage}] cached={len(results)} to_process={len(to_process)}")

    if not to_process:
        return results

    sem = asyncio.Semaphore(concurrency)
    total_prompt = 0
    total_comp = 0
    done_count = 0
    start_time = time.time()

    async def process_one(item, lang, session):
        nonlocal total_prompt, total_comp, done_count
        async with sem:
            prompt = build_correction_prompt(item, lang, model)
            text, pt, ct = await call_api(session, model, prompt)
            total_prompt += pt
            total_comp += ct
            done_count += 1
            corrected = parse_translation_response(text) if text else None
            entry = {
                "item_id": item["item_id"],
                "lang_code": lang,
                "model": model,
                "stage": stage,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "corrected_fields": corrected or {},
                "parse_success": corrected is not None,
                "prompt_tokens": pt,
                "completion_tokens": ct,
                "raw_response_head": text[:300] if text else None,
            }
            key = (item["item_id"], lang, model)
            save_cache_entry(cache_file, entry)
            cache[key] = entry
            results[(item["item_id"], lang)] = corrected or {}
            status = "✓" if corrected else "✗ parse_fail"
            if done_count % 50 == 0 or not corrected:
                logger.info(
                    f"[{stage}] {done_count}/{len(to_process)} | "
                    f"{item['item_id']} [{lang}] {status} | "
                    f"tokens: p={total_prompt:,} c={total_comp:,}"
                )

    async with httpx.AsyncClient() as session:
        tasks = [process_one(item, lang, session) for item, lang in to_process]
        await asyncio.gather(*tasks)

    elapsed = time.time() - start_time
    log_tokens(token_log, model, stage, len(to_process),
               total_prompt, total_comp, elapsed)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Apply correction with rollback safety
# ─────────────────────────────────────────────────────────────────────────────
def apply_correction(item, corrected_fields, lang, model_name):
    orig_passage = item.get("flores_passage", "")
    new_passage = corrected_fields.get("flores_passage", "")

    updated = dict(item)
    correction_applied = False
    rollback = False

    # A valid correction must have non-empty passage and question
    if (corrected_fields and new_passage and len(new_passage.strip()) > 10 
            and corrected_fields.get("question", "").strip()):
        for field in FIELDS_TO_TRANSLATE:
            val = corrected_fields.get(field, "")
            if val:
                updated[field] = val
        correction_applied = True
    elif corrected_fields:
        rollback = True
        logger.warning(f"ROLLBACK {item['item_id']} [{lang}]: invalid/empty fields from {model_name}")

    meta = {
        "item_id": item["item_id"],
        "lang_code": lang,
        "correction_model": model_name,
        "correction_applied": correction_applied,
        "rollback": rollback,
        "parse_success": bool(corrected_fields),
    }
    return updated, meta


# ─────────────────────────────────────────────────────────────────────────────
# LaBSE re-scoring
# ─────────────────────────────────────────────────────────────────────────────
def labse_rescore(items, labse_path, batch_size=64):
    from sentence_transformers import SentenceTransformer
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"LaBSE re-scoring on {device} ({len(items)} items)...")
    model = SentenceTransformer(labse_path, device=device)
    pairs_en  = [it.get("flores_passage_en", it.get("flores_passage", "")) for it in items]
    pairs_tgt = [it.get("flores_passage", "") for it in items]
    all_scores = []
    for i in range(0, len(pairs_en), batch_size):
        emb_en  = model.encode(pairs_en[i:i+batch_size],  convert_to_tensor=True, normalize_embeddings=True)
        emb_tgt = model.encode(pairs_tgt[i:i+batch_size], convert_to_tensor=True, normalize_embeddings=True)
        all_scores.extend((emb_en * emb_tgt).sum(dim=1).cpu().tolist())
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return all_scores


# ─────────────────────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--translated-dir",  default="translated")
    p.add_argument("--output-dir",      default="corrected")
    p.add_argument("--cache-file",      default="correction_cache.jsonl")
    p.add_argument("--token-log",       default="correction_token_usage.jsonl")
    p.add_argument("--correction-log",  default="correction_log.jsonl")
    p.add_argument("--labse-path",      default="sentence-transformers/LaBSE")
    p.add_argument("--languages",       nargs="+",
                   default=["ory_Orya", "asm_Beng", "sin_Sinh", "mlt_Latn"])
    p.add_argument("--gemini-concurrency", type=int, default=20)
    p.add_argument("--qwen-concurrency",   type=int, default=10)
    p.add_argument("--pilot-n",         type=int, default=0,
                   help="If >0, only process first N items per language (pilot mode)")
    p.add_argument("--skip-stage1",     action="store_true")
    p.add_argument("--skip-stage2",     action="store_true")
    p.add_argument("--rescore-only",    action="store_true")
    p.add_argument("--no-labse",        action="store_true")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

import requests

def run_openrouter_batch(items_langs, cache, cache_file, token_log, stage="stage1_gemini"):
    """Submit ALL OpenRouter Batch API jobs simultaneously, then poll and unpack all results."""
    to_process = []
    results = {}
    for item, lang in items_langs:
        key = (item["item_id"], lang, GEMINI_MODEL)
        if key in cache:
            results[(item["item_id"], lang)] = cache[key].get("corrected_fields", {})
        else:
            to_process.append((item, lang))

    logger.info(f"[{stage}] {len(results)} cached | {len(to_process)} to submit to OpenRouter Batch API")
    if not to_process:
        return results

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    requests_data = []
    for item, lang in to_process:
        custom_id = f"{item['item_id']}|{lang}"
        prompt = build_correction_prompt(item, lang, GEMINI_MODEL)
        requests_data.append({
            "custom_id": custom_id,
            "body": {
                "model": GEMINI_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 4096,
                "temperature": 0.1,
            }
        })

    # STEP 1: Submit ALL batch chunks SIMULTANEOUSLY
    batch_size = 1000
    submitted_batches = []  # [(batch_id, chunk_size, start_time)]

    for b_idx in range(0, len(requests_data), batch_size):
        chunk = requests_data[b_idx:b_idx+batch_size]
        payload = {
            "endpoint": "/v1/chat/completions",
            "model": GEMINI_MODEL,
            "requests": chunk,
        }
        b_num = b_idx // batch_size + 1
        logger.info(f"[{stage}] Submitting Batch {b_num} ({len(chunk)} requests)...")
        res = requests.post("https://openrouter.ai/api/beta/batches", headers=headers, json=payload)
        if res.status_code not in (200, 202):
            logger.error(f"Batch {b_num} submit failed (HTTP {res.status_code}): {res.text[:300]}")
            continue

        b_info = res.json()
        batch_id = b_info.get("id")
        logger.info(f"[{stage}] Batch {b_num} submitted successfully! ID: {batch_id}")
        submitted_batches.append((batch_id, len(chunk), time.time()))

    logger.info(f"[{stage}] All {len(submitted_batches)} batches submitted simultaneously! Polling for completion...")

    # STEP 2: Poll all submitted batches until completed
    pending = list(submitted_batches)
    while pending:
        time.sleep(10)
        still_pending = []
        for batch_id, chunk_len, t0 in pending:
            poll_url = f"https://openrouter.ai/api/beta/batches/{batch_id}"
            try:
                poll_res = requests.get(poll_url, headers=headers)
                if poll_res.status_code != 200:
                    still_pending.append((batch_id, chunk_len, t0))
                    continue
                data = poll_res.json()
                status = data.get("status")
                req_counts = data.get("request_counts", {})
                completed = req_counts.get("completed", 0)
                total = req_counts.get("total", chunk_len)

                logger.info(f"[{stage}] Batch {batch_id[-12:]}: status={status} ({completed}/{total} completed)")

                if status in ["completed", "failed", "cancelled", "expired"]:
                    # Batch finished — unpack results
                    batch_results = data.get("results") or []
                    logger.info(f"[{stage}] Batch {batch_id[-12:]} finished with status '{status}'. Unpacking {len(batch_results)} results...")
                    tot_p, tot_c = 0, 0
                    for item_res in batch_results:
                        if not isinstance(item_res, dict):
                            continue
                        cid = item_res.get("custom_id", "")
                        if "|" not in cid:
                            continue
                        item_id, lang = cid.split("|", 1)
                        resp = item_res.get("response") or {}
                        resp_body = resp.get("body") or {}
                        text = ""
                        usage = {}
                        try:
                            text = resp_body["choices"][0]["message"]["content"]
                            usage = resp_body.get("usage") or {}
                            tot_p += usage.get("prompt_tokens", 0)
                            tot_c += usage.get("completion_tokens", 0)
                        except Exception:
                            pass

                        corrected = parse_translation_response(text) if text else None
                        entry = {
                            "item_id": item_id,
                            "lang_code": lang,
                            "model": GEMINI_MODEL,
                            "stage": stage,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "corrected_fields": corrected or {},
                            "parse_success": corrected is not None,
                            "prompt_tokens": usage.get("prompt_tokens", 0) if text else 0,
                            "completion_tokens": usage.get("completion_tokens", 0) if text else 0,
                        }
                        key = (item_id, lang, GEMINI_MODEL)
                        save_cache_entry(cache_file, entry)
                        cache[key] = entry
                        results[(item_id, lang)] = corrected or {}

                    log_tokens(token_log, GEMINI_MODEL, stage, chunk_len, tot_p, tot_c, time.time() - t0)
                else:
                    still_pending.append((batch_id, chunk_len, t0))
            except Exception as ex:
                logger.warning(f"Poll error for {batch_id}: {ex}")
                still_pending.append((batch_id, chunk_len, t0))

        pending = still_pending

    return results



async def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("=" * 70)
    logger.info("TRANSLATION QUALITY CORRECTION PIPELINE")
    logger.info(f"  Mode   : {'PILOT (n=' + str(args.pilot_n) + ' per lang)' if args.pilot_n else 'FULL RUN (3600 items)'}")
    logger.info(f"  Stage1 : {GEMINI_MODEL}")
    logger.info(f"  Stage2 : {QWEN_MODEL}")
    logger.info(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 70)

    cache = load_cache(args.cache_file)
    logger.info(f"Cache loaded: {len(cache)} entries")

    # Load all items
    all_items = {}
    for lang in args.languages:
        f = os.path.join(args.translated_dir, f"translated_{lang}.jsonl")
        items = [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
        if args.pilot_n:
            items = items[:args.pilot_n]
        all_items[lang] = items
        logger.info(f"  [{lang}] {len(items)} items loaded")

    # ── Stage 1: Gemini ───────────────────────────────────────────────────────
    gemini_results = {}
    if not args.skip_stage1 and not args.rescore_only:
        stage1_pairs = []
        for lang, items in all_items.items():
            for item in items:
                stage1_pairs.append((item, lang))

        logger.info(f"\n{'='*70}")
        logger.info(f"STAGE 1: {GEMINI_MODEL}  ({len(stage1_pairs)} items)")
        logger.info(f"{'='*70}")

        gemini_results = run_openrouter_batch(
            stage1_pairs, cache, args.cache_file, args.token_log, "stage1_gemini"
        )

        if args.pilot_n:
            logger.info("\n=== PILOT SAMPLE (Stage 1 - Gemini) ===")
            for (iid, lc), fields in list(gemini_results.items())[:5]:
                logger.info(f"  [{iid}][{lc}] -> {str(fields.get('flores_passage', ''))[:120]}")
            logger.info("Pilot Stage 1 complete. Proceeding automatically..."); answer = "y"
            if answer != "y":
                logger.info("Stopping after pilot.")
                return

    # Apply Stage 1 + identify Stage 2 candidates
    stage1_meta = {}
    stage2_candidates = []

    for lang, items in all_items.items():
        for idx, item in enumerate(items):
            key = (item["item_id"], lang)
            cf = gemini_results.get(key, {})
            updated, meta = apply_correction(item, cf, lang, GEMINI_MODEL)
            stage1_meta[key] = meta
            all_items[lang][idx] = updated

            # Stage 2 candidates: parse fail, still low chrF, or hard asm_Beng
            chrf_after = meta.get("new_chrf_passage", meta.get("orig_chrf_passage", 100))
            if (not meta["parse_success"]
                    or chrf_after < 50
                    or (lang == "asm_Beng" and item.get("labse_a", 1.0) < 0.80)):
                stage2_candidates.append((updated, lang))

    logger.info(f"\nStage 1 done. Stage 2 candidates: {len(stage2_candidates)}")

    # ── Stage 2: Qwen ─────────────────────────────────────────────────────────
    qwen_results = {}
    if not args.skip_stage2 and not args.rescore_only and stage2_candidates:
        logger.info(f"\n{'='*70}")
        logger.info(f"STAGE 2: {QWEN_MODEL}  ({len(stage2_candidates)} items)")
        logger.info(f"{'='*70}")
        qwen_results = await correct_batch(
            stage2_candidates, QWEN_MODEL, cache,
            args.cache_file, args.token_log,
            "stage2_qwen", args.qwen_concurrency,
        )

        for lang, items in all_items.items():
            for idx, item in enumerate(items):
                key = (item["item_id"], lang)
                cf = qwen_results.get(key, {})
                if not cf:
                    continue
                updated, meta2 = apply_correction(item, cf, lang, QWEN_MODEL)
                if meta2["correction_applied"]:
                    all_items[lang][idx] = updated
                    stage1_meta[key]["stage2_model"] = QWEN_MODEL
                    stage1_meta[key]["stage2_applied"] = True
                    stage1_meta[key]["stage2_chrf"] = None

    # ── Stage 3: Re-score ─────────────────────────────────────────────────────
    logger.info(f"\n{'='*70}")
    logger.info("STAGE 3: Re-scoring (chrF++ + LaBSE)")
    logger.info(f"{'='*70}")

    correction_log_entries = []
    final_stats = {}

    for lang, items in all_items.items():
        # chrF++ for all items
        for item in items:
            en  = item.get("flores_passage_en", item.get("flores_passage", ""))
            tgt = item.get("flores_passage", "")
            item["corrected_chrf"] = round(chrf_score(tgt, en), 3)

        # LaBSE
        if not args.no_labse and os.path.isdir(args.labse_path):
            labse_scores = labse_rescore(items, args.labse_path)
            for item, score in zip(items, labse_scores):
                item["corrected_labse"] = round(score, 4)
        else:
            for item in items:
                item["corrected_labse"] = None

        orig_chrfs = [it.get("chrf_a", 0) for it in items]
        new_chrfs  = [it.get("corrected_chrf", 0) for it in items]
        improved  = sum(1 for o, n in zip(orig_chrfs, new_chrfs) if n > o + 1.0)
        degraded  = sum(1 for o, n in zip(orig_chrfs, new_chrfs) if n < o - 1.0)

        final_stats[lang] = {
            "n": len(items),
            "orig_chrf_mean": round(sum(orig_chrfs)/len(orig_chrfs), 2),
            "new_chrf_mean":  round(sum(new_chrfs)/len(new_chrfs), 2),
            "chrf_delta":     round(sum(new_chrfs)/len(new_chrfs) - sum(orig_chrfs)/len(orig_chrfs), 2),
            "items_improved": improved,
            "items_degraded": degraded,
            "items_rolled_back": sum(1 for m in stage1_meta.values()
                                     if m.get("lang_code") == lang and m.get("rollback")),
        }
        if not args.no_labse:
            orig_l = [it.get("labse_a", 0) for it in items]
            new_l  = [it.get("corrected_labse") or it.get("labse_a", 0) for it in items]
            final_stats[lang]["orig_labse_mean"] = round(sum(orig_l)/len(orig_l), 4)
            final_stats[lang]["new_labse_mean"]  = round(sum(new_l)/len(new_l), 4)
            final_stats[lang]["labse_delta"]     = round(sum(new_l)/len(new_l) - sum(orig_l)/len(orig_l), 4)

        s = final_stats[lang]
        logger.info(
            f"[{lang}] chrF: {s['orig_chrf_mean']} → {s['new_chrf_mean']} "
            f"(Δ{s['chrf_delta']:+.2f}) | improved={improved} | degraded={degraded}"
        )

        # Write corrected JSONL
        out_path = os.path.join(args.output_dir, f"corrected_{lang}.jsonl")
        with open(out_path, "w", encoding="utf-8") as fout:
            for item in items:
                key = (item["item_id"], lang)
                m = stage1_meta.get(key, {})
                item["correction_applied"]  = m.get("correction_applied", False)
                item["correction_model"]    = m.get("correction_model", "none")
                item["stage2_applied"]      = m.get("stage2_applied", False)
                item["rollback"]            = m.get("rollback", False)
                fout.write(json.dumps(item, ensure_ascii=False) + "\n")
        logger.info(f"[{lang}] → {out_path} ({len(items)} records)")

        for item in items:
            key = (item["item_id"], lang)
            m = stage1_meta.get(key, {})
            correction_log_entries.append({
                "item_id":           item["item_id"],
                "lang_code":         lang,
                "orig_chrf":         m.get("orig_chrf_passage", 0),
                "new_chrf":          item.get("corrected_chrf", 0),
                "chrf_delta":        round(item.get("corrected_chrf", 0) - m.get("orig_chrf_passage", 0), 3),
                "orig_labse":        item.get("labse_a", 0),
                "new_labse":         item.get("corrected_labse"),
                "correction_applied": m.get("correction_applied", False),
                "rollback":          m.get("rollback", False),
                "stage2_applied":    m.get("stage2_applied", False),
                "correction_model":  m.get("correction_model", "none"),
            })

    # Write correction log
    with open(args.correction_log, "w", encoding="utf-8") as f:
        for e in correction_log_entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    logger.info(f"Per-item correction log → {args.correction_log}")

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 70)
    header = f"{'Language':<12} {'n':>5} {'chrF_orig':>10} {'chrF_new':>10} {'Δ':>8} {'Impr':>6} {'Degr':>6}"
    logger.info(header)
    logger.info("-" * 70)
    for lang, s in final_stats.items():
        logger.info(
            f"{lang:<12} {s['n']:>5} {s['orig_chrf_mean']:>10.2f} "
            f"{s['new_chrf_mean']:>10.2f} {s['chrf_delta']:>+8.2f} "
            f"{s['items_improved']:>6} {s['items_degraded']:>6}"
        )

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "models_used": {"stage1": GEMINI_MODEL, "stage2": QWEN_MODEL},
        "total_items": sum(s["n"] for s in final_stats.values()),
        "per_language": final_stats,
    }
    summary_path = os.path.join(args.output_dir, "correction_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"\nSummary → {summary_path}")
    logger.info("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
