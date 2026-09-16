#!/usr/bin/env python3
"""
apply_corrections.py  [GPU NODE]
=================================
Reads correction_cache.jsonl (produced by quality_correction.py on login node),
applies the corrected translations (with rollback safety), re-scores with
LaBSE + BGE-M3 + Qwen3-Embedding, and writes final corrected_*.jsonl files.

No API calls. No internet required. Pure GPU/local work.

Usage:
  python apply_corrections.py \
      --translated-dir translated \
      --cache-file     correction_cache.jsonl \
      --output-dir     corrected
"""

import os, gc, sys, json, math, time, argparse, logging
from datetime import datetime, timezone

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("apply_corrections.log", mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger("apply")

MODEL_DIR = os.environ.get("MODEL_DIR", "models")
LANG_NAMES = {
    "ory_Orya": "Odia", "asm_Beng": "Assamese",
    "sin_Sinh": "Sinhala", "mlt_Latn": "Maltese",
}
FIELDS_TO_TRANSLATE = [
    "flores_passage", "question",
    "mc_answer1", "mc_answer2", "mc_answer3", "mc_answer4",
]


# ── chrF++ ───────────────────────────────────────────────────────────────────
def _char_ngrams(text, n):
    ng = {}
    for i in range(len(text)-n+1):
        k = text[i:i+n]; ng[k] = ng.get(k,0)+1
    return ng

def chrf_score(hyp, ref, max_n=6, beta=2.0):
    if not hyp or not ref: return 0.0
    precs, recs = [], []
    for n in range(1, max_n+1):
        h = _char_ngrams(hyp, n); r = _char_ngrams(ref, n)
        if not h or not r: continue
        ov = sum(min(h.get(k,0), r.get(k,0)) for k in set(h)&set(r))
        precs.append(ov/sum(h.values())); recs.append(ov/sum(r.values()))
    if not precs: return 0.0
    p = sum(precs)/len(precs); r_ = sum(recs)/len(recs)
    if p+r_==0: return 0.0
    return (1+beta**2)*p*r_/(beta**2*p+r_)*100.0

def bleu_score(hyp, ref, max_n=4):
    import re, math
    if not hyp or not ref: return 0.0
    ht = re.findall(r'\S+', hyp.lower()); rt = re.findall(r'\S+', ref.lower())
    if not ht: return 0.0
    bp = min(0.0, 1.0 - len(rt)/len(ht))
    log_ps = []
    for n in range(1, max_n+1):
        def ng(toks, n):
            d={}
            for i in range(len(toks)-n+1):
                k=tuple(toks[i:i+n]); d[k]=d.get(k,0)+1
            return d
        h=ng(ht,n); r=ng(rt,n)
        if not h: break
        cl=sum(min(v,r.get(k,0)) for k,v in h.items()); tot=sum(h.values())
        if cl==0: return 0.0
        log_ps.append(math.log(cl/tot))
    if not log_ps: return 0.0
    return math.exp(bp + sum(log_ps)/len(log_ps))*100.0

def meteor_simple(hyp, ref):
    import re
    if not hyp or not ref: return 0.0
    h=set(re.findall(r'\S+',hyp.lower())); r=set(re.findall(r'\S+',ref.lower()))
    if not h or not r: return 0.0
    m=len(h&r)
    if m==0: return 0.0
    p=m/len(h); rc=m/len(r); b=0.5
    return (1+b**2)*p*rc/(b**2*p+rc)*100.0


# ── Embedding similarity ──────────────────────────────────────────────────────
def embed_cosine(items, model_path, label, batch_size=64, normalize=True):
    from sentence_transformers import SentenceTransformer
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"  [{label}] loading on {device}...")
    model = SentenceTransformer(model_path, device=device, trust_remote_code=True)
    en_texts  = [it.get("flores_passage_en", it.get("flores_passage","")) for it in items]
    tgt_texts = [it.get("flores_passage","") for it in items]
    scores = []
    for i in range(0, len(en_texts), batch_size):
        emb_en  = model.encode(en_texts[i:i+batch_size],
                               convert_to_tensor=True, normalize_embeddings=normalize)
        emb_tgt = model.encode(tgt_texts[i:i+batch_size],
                               convert_to_tensor=True, normalize_embeddings=normalize)
        scores.extend((emb_en * emb_tgt).sum(dim=1).cpu().tolist())
        if (i // batch_size) % 4 == 0:
            logger.info(f"  [{label}] {min(i+batch_size,len(en_texts))}/{len(en_texts)}")
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return scores


# ── Load cache ────────────────────────────────────────────────────────────────
def load_cache(cache_file):
    cache = {}  # (item_id, lang_code) -> best corrected_fields
    model_priority = {
        "stage2_qwen": 2,    # Qwen Stage 2 > Gemini Stage 1
        "stage1_gemini": 1,
    }
    if not os.path.exists(cache_file):
        logger.error(f"Cache file not found: {cache_file}")
        return cache

    with open(cache_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                entry = json.loads(line)
                if not entry.get("parse_success"): continue
                cf = entry.get("corrected_fields", {})
                if not cf: continue
                key = (entry["item_id"], entry["lang_code"])
                stage = entry.get("stage", "")
                priority = model_priority.get(stage, 0)
                # Keep highest-priority (Stage 2 > Stage 1)
                if key not in cache or priority > cache[key]["_priority"]:
                    cache[key] = {**cf, "_priority": priority,
                                  "_model": entry.get("model",""),
                                  "_stage": stage}
            except Exception:
                pass

    logger.info(f"Cache loaded: {len(cache)} correctable items")
    return cache


# ── Apply correction with rollback ────────────────────────────────────────────
def apply_one(item, corrected_fields, lang_code):
    orig = item.get("flores_passage", "")
    new  = corrected_fields.get("flores_passage", "")

    updated = dict(item)
    applied = False
    rollback = False
    model_used = corrected_fields.get("_model", "none")
    stage_used = corrected_fields.get("_stage", "none")

    if corrected_fields and new and len(new.strip()) > 10 and corrected_fields.get("question", "").strip():
        for field in FIELDS_TO_TRANSLATE:
            v = corrected_fields.get(field, "")
            if v:
                updated[field] = v
        applied = True
    elif corrected_fields:
        rollback = True
        logger.warning(f"ROLLBACK {item['item_id']} [{lang_code}]: invalid/empty fields from {model_used}")

    updated["correction_applied"]  = applied
    updated["correction_rollback"] = rollback
    updated["correction_model"]    = model_used
    updated["correction_stage"]    = stage_used

    return updated, {
        "item_id": item["item_id"],
        "lang_code": lang_code,
        "applied": applied,
        "rollback": rollback,
        "model": model_used,
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--translated-dir", default="translated")
    p.add_argument("--cache-file",     default="correction_cache.jsonl")
    p.add_argument("--output-dir",     default="corrected")
    p.add_argument("--correction-log", default="correction_log.jsonl")
    p.add_argument("--labse-path",     default="sentence-transformers/LaBSE")
    p.add_argument("--bge-path",       default=f"{MODEL_DIR}/bge-m3")
    p.add_argument("--qwen-emb-path",  default=f"{MODEL_DIR}/Qwen3-Embedding-0.6B")
    p.add_argument("--languages", nargs="+",
                   default=["ory_Orya","asm_Beng","sin_Sinh","mlt_Latn"])
    p.add_argument("--skip-bge",       action="store_true")
    p.add_argument("--skip-qwen-emb",  action="store_true")
    p.add_argument("--batch-size",     type=int, default=64)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("=" * 70)
    logger.info("APPLY CORRECTIONS + GPU SCORING  [GPU NODE]")
    logger.info(f"  Cache    : {args.cache_file}")
    logger.info(f"  Input    : {args.translated_dir}")
    logger.info(f"  Output   : {args.output_dir}")
    logger.info(f"  Started  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 70)

    cache = load_cache(args.cache_file)

    correction_log = []
    final_stats = {}

    for lang in args.languages:
        f = os.path.join(args.translated_dir, f"translated_{lang}.jsonl")
        items = [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
        logger.info(f"\n[{lang}] {len(items)} items loaded")

        # Apply corrections
        updated_items = []
        lang_log = []
        for item in items:
            key = (item["item_id"], lang)
            cf = cache.get(key, {})
            upd, meta = apply_one(item, cf, lang)
            updated_items.append(upd)
            lang_log.append(meta)
        correction_log.extend(lang_log)

        applied = sum(1 for m in lang_log if m["applied"])
        rolled  = sum(1 for m in lang_log if m["rollback"])
        logger.info(f"  Applied: {applied} | Rolled back: {rolled} | "
                    f"No cache: {len(items)-applied-rolled}")

        # ── Tier 1: chrF++, BLEU, METEOR ──────────────────────────────────
        for item in updated_items:
            en  = item.get("flores_passage_en", item.get("flores_passage",""))
            tgt = item.get("flores_passage","")
            item["corrected_chrf"]   = round(chrf_score(tgt, en), 3)
            item["corrected_bleu"]   = round(bleu_score(tgt, en), 3)
            item["corrected_meteor"] = round(meteor_simple(tgt, en), 3)

        # ── Tier 2: LaBSE ─────────────────────────────────────────────────
        if os.path.isdir(args.labse_path):
            logger.info(f"  [LaBSE] scoring...")
            labse_scores = embed_cosine(updated_items, args.labse_path,
                                        "LaBSE", args.batch_size)
            for item, score in zip(updated_items, labse_scores):
                item["corrected_labse"] = round(score, 4)
        else:
            logger.warning(f"  LaBSE not found at {args.labse_path}")
            for item in updated_items:
                item["corrected_labse"] = None

        # ── Tier 2: BGE-M3 ────────────────────────────────────────────────
        if not args.skip_bge and os.path.isdir(args.bge_path):
            logger.info(f"  [BGE-M3] scoring...")
            bge_scores = embed_cosine(updated_items, args.bge_path,
                                      "BGE-M3", args.batch_size//2)
            for item, score in zip(updated_items, bge_scores):
                item["corrected_bge_m3"] = round(score, 4)
        else:
            for item in updated_items:
                item["corrected_bge_m3"] = None

        # ── Tier 2: Qwen3-Embedding ───────────────────────────────────────
        if not args.skip_qwen_emb and os.path.isdir(args.qwen_emb_path):
            logger.info(f"  [Qwen3-Embedding] scoring...")
            qe_scores = embed_cosine(updated_items, args.qwen_emb_path,
                                     "Qwen3-Embedding", args.batch_size//2)
            for item, score in zip(updated_items, qe_scores):
                item["corrected_qwen_emb"] = round(score, 4)
        else:
            for item in updated_items:
                item["corrected_qwen_emb"] = None

        # ── Stats ──────────────────────────────────────────────────────────
        orig_chrf = [it.get("chrf_a", 0) for it in updated_items]
        new_chrf  = [it.get("corrected_chrf", 0) for it in updated_items]
        orig_labse= [it.get("labse_a", 0) for it in updated_items]
        new_labse = [it.get("corrected_labse") or it.get("labse_a",0) for it in updated_items]

        final_stats[lang] = {
            "n": len(updated_items),
            "corrections_applied": applied,
            "rollbacks": rolled,
            "chrf_before": round(sum(orig_chrf)/len(orig_chrf), 3),
            "chrf_after":  round(sum(new_chrf)/len(new_chrf), 3),
            "chrf_delta":  round(sum(new_chrf)/len(new_chrf) - sum(orig_chrf)/len(orig_chrf), 3),
            "labse_before":round(sum(orig_labse)/len(orig_labse), 4),
            "labse_after": round(sum(new_labse)/len(new_labse), 4),
            "labse_delta": round(sum(new_labse)/len(new_labse) - sum(orig_labse)/len(orig_labse), 4),
        }

        # ── Write corrected JSONL ──────────────────────────────────────────
        out = os.path.join(args.output_dir, f"corrected_{lang}.jsonl")
        with open(out, "w", encoding="utf-8") as fout:
            for item in updated_items:
                fout.write(json.dumps(item, ensure_ascii=False) + "\n")
        s = final_stats[lang]
        logger.info(
            f"  [{lang}] chrF: {s['chrf_before']}→{s['chrf_after']} "
            f"(Δ{s['chrf_delta']:+.3f}) | "
            f"LaBSE: {s['labse_before']}→{s['labse_after']} "
            f"(Δ{s['labse_delta']:+.4f})"
        )
        logger.info(f"  Written: {out}")

    # Write correction log
    with open(args.correction_log, "w", encoding="utf-8") as f:
        for e in correction_log:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    # Save stats JSON
    stats_path = os.path.join(args.output_dir, "apply_corrections_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "per_language": final_stats,
            "total_applied": sum(s["corrections_applied"] for s in final_stats.values()),
            "total_rollbacks": sum(s["rollbacks"] for s in final_stats.values()),
        }, f, indent=2, ensure_ascii=False)

    # Print summary table
    logger.info("\n" + "="*70)
    logger.info("APPLY CORRECTIONS SUMMARY")
    logger.info("="*70)
    logger.info(f"{'Lang':<12} {'n':>5} {'Applied':>8} {'Roll':>6} "
                f"{'chrF_bef':>10} {'chrF_aft':>10} {'Δ':>8} "
                f"{'LaBSE_bef':>10} {'LaBSE_aft':>10} {'Δ':>8}")
    logger.info("-"*90)
    for lang, s in final_stats.items():
        logger.info(
            f"{lang:<12} {s['n']:>5} {s['corrections_applied']:>8} {s['rollbacks']:>6} "
            f"{s['chrf_before']:>10.3f} {s['chrf_after']:>10.3f} {s['chrf_delta']:>+8.3f} "
            f"{s['labse_before']:>10.4f} {s['labse_after']:>10.4f} {s['labse_delta']:>+8.4f}"
        )
    logger.info(f"\nStats JSON → {stats_path}")
    logger.info(f"Correction log → {args.correction_log}")
    logger.info("="*70)


if __name__ == "__main__":
    main()
