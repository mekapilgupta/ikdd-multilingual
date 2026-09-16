#!/usr/bin/env python3
"""
evaluate_quality.py — Research-Grade Translation Quality Evaluation
====================================================================
Metrics computed:
  Tier 1 (fast, always):    chrF++, BLEU, METEOR
  Tier 2 (GPU embeddings):  LaBSE, BGE-M3, Qwen3-Embedding-0.6B
  Tier 3 (GPU neural):      BERTScore (multilingual), COMET-22

Outputs:
  ikdd/eval_results/eval_{tag}.json         — full per-item scores
  ikdd/eval_results/eval_{tag}_summary.json — per-language aggregate tables
  ikdd/eval_results/eval_{tag}_report.md   — human-readable markdown report

Usage:
  # Before correction:
  python ikdd/evaluate_quality.py --input-dir ikdd/translated --tag before_correction

  # After correction:
  python ikdd/evaluate_quality.py --input-dir ikdd/corrected --tag after_correction

  # Skip slow GPU metrics for quick check:
  python ikdd/evaluate_quality.py --input-dir ikdd/translated --tag quick --skip-tier3

  # Generate comparison report (run after both before and after):
  python ikdd/evaluate_quality.py --compare \
      --before ikdd/eval_results/eval_before_correction_summary.json \
      --after  ikdd/eval_results/eval_after_correction_summary.json
"""

import os, gc, sys, json, time, argparse, logging, math
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("ikdd/evaluation.log", mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger("eval")

LANG_NAMES = {
    "ory_Orya": "Odia",
    "asm_Beng": "Assamese",
    "sin_Sinh": "Sinhala",
    "mlt_Latn": "Maltese",
}

MODEL_DIR = os.environ.get("MODEL_DIR", "models")


# ─────────────────────────────────────────────────────────────────────────────
# Tier 1: Fast metrics (pure Python / CPU)
# ─────────────────────────────────────────────────────────────────────────────

def _char_ngrams(text: str, n: int) -> dict:
    ng = {}
    for i in range(len(text) - n + 1):
        k = text[i:i+n]; ng[k] = ng.get(k, 0) + 1
    return ng

def chrf_score(hyp: str, ref: str, max_n: int = 6, beta: float = 2.0) -> float:
    if not hyp or not ref: return 0.0
    precs, recs = [], []
    for n in range(1, max_n + 1):
        h = _char_ngrams(hyp, n); r = _char_ngrams(ref, n)
        if not h or not r: continue
        overlap = sum(min(h.get(k,0), r.get(k,0)) for k in set(h)&set(r))
        precs.append(overlap / sum(h.values()))
        recs.append(overlap / sum(r.values()))
    if not precs: return 0.0
    p = sum(precs)/len(precs); r_ = sum(recs)/len(recs)
    if p + r_ == 0: return 0.0
    return (1+beta**2)*p*r_/(beta**2*p+r_)*100.0

def bleu_score(hyp: str, ref: str, max_n: int = 4) -> float:
    """Sentence BLEU (micro-average, no smoothing)."""
    if not hyp or not ref: return 0.0
    import re
    hyp_tok = re.findall(r'\S+', hyp.lower())
    ref_tok = re.findall(r'\S+', ref.lower())
    if not hyp_tok: return 0.0
    log_bp = min(0.0, 1.0 - len(ref_tok)/len(hyp_tok))
    log_precs = []
    for n in range(1, max_n + 1):
        def ngrams(toks, n):
            ng = {}
            for i in range(len(toks)-n+1):
                k = tuple(toks[i:i+n]); ng[k] = ng.get(k,0)+1
            return ng
        h = ngrams(hyp_tok, n); r = ngrams(ref_tok, n)
        if not h: break
        clipped = sum(min(v, r.get(k,0)) for k,v in h.items())
        total = sum(h.values())
        if clipped == 0: return 0.0
        log_precs.append(math.log(clipped/total))
    if not log_precs: return 0.0
    avg_log = sum(log_precs)/len(log_precs)
    return math.exp(log_bp + avg_log) * 100.0

def meteor_score_simple(hyp: str, ref: str) -> float:
    """Simplified METEOR (unigram F-measure with recall weighting β=0.5)."""
    import re
    if not hyp or not ref: return 0.0
    h = set(re.findall(r'\S+', hyp.lower()))
    r = set(re.findall(r'\S+', ref.lower()))
    if not h or not r: return 0.0
    match = len(h & r)
    if match == 0: return 0.0
    p = match / len(h); r_ = match / len(r)
    beta = 0.5
    return (1 + beta**2) * p * r_ / (beta**2 * p + r_) * 100.0

def compute_tier1(items: list) -> list:
    """Returns list of per-item metric dicts."""
    results = []
    for it in items:
        en  = it.get("flores_passage_en", it.get("flores_passage", ""))
        tgt = it.get("flores_passage", "")
        results.append({
            "item_id":  it["item_id"],
            "lang_code": it["lang_code"],
            "chrf":    round(chrf_score(tgt, en), 3),
            "bleu":    round(bleu_score(tgt, en), 3),
            "meteor":  round(meteor_score_simple(tgt, en), 3),
        })
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Tier 2: GPU embedding metrics
# ─────────────────────────────────────────────────────────────────────────────

def embed_similarity(items: list, model_path_or_name: str,
                     model_label: str, batch_size: int = 64) -> list:
    """Cosine similarity between EN passage and target passage."""
    from sentence_transformers import SentenceTransformer
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"  Loading {model_label} on {device}...")
    model = SentenceTransformer(model_path_or_name, device=device,
                                trust_remote_code=True)
    pairs_en  = [it.get("flores_passage_en", it.get("flores_passage","")) for it in items]
    pairs_tgt = [it.get("flores_passage","") for it in items]
    scores = []
    for i in range(0, len(pairs_en), batch_size):
        emb_en  = model.encode(pairs_en[i:i+batch_size],
                               convert_to_tensor=True, normalize_embeddings=True)
        emb_tgt = model.encode(pairs_tgt[i:i+batch_size],
                               convert_to_tensor=True, normalize_embeddings=True)
        scores.extend((emb_en * emb_tgt).sum(dim=1).cpu().tolist())
        if (i//batch_size) % 5 == 0:
            logger.info(f"    {model_label}: {min(i+batch_size, len(pairs_en))}/{len(pairs_en)}")
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return scores

def compute_tier2(items: list, args) -> dict:
    """Returns {metric_name: [scores]}."""
    results = {}

    # LaBSE
    if os.path.isdir(args.labse_path):
        results["labse"] = embed_similarity(items, args.labse_path, "LaBSE")
    else:
        logger.warning(f"LaBSE not found at {args.labse_path}, skipping")

    # BGE-M3
    bge_path = os.path.join(MODEL_DIR, "bge-m3")
    if os.path.isdir(bge_path):
        results["bge_m3"] = embed_similarity(items, bge_path, "BGE-M3", batch_size=32)
    elif not args.skip_download:
        logger.info("  BGE-M3 not found locally — downloading BAAI/bge-m3...")
        results["bge_m3"] = embed_similarity(items, "BAAI/bge-m3", "BGE-M3", batch_size=32)
    else:
        logger.warning("BGE-M3 not found, skipping")

    # Qwen3-Embedding-0.6B
    qwen_emb_path = os.path.join(MODEL_DIR, "Qwen3-Embedding-0.6B")
    if os.path.isdir(qwen_emb_path):
        results["qwen_embedding"] = embed_similarity(
            items, qwen_emb_path, "Qwen3-Embedding-0.6B", batch_size=32)
    elif not args.skip_download:
        logger.info("  Qwen3-Embedding not found locally — downloading...")
        results["qwen_embedding"] = embed_similarity(
            items, "Qwen/Qwen3-Embedding-0.6B", "Qwen3-Embedding-0.6B", batch_size=32)
    else:
        logger.warning("Qwen3-Embedding not found, skipping")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Tier 3: Neural MT evaluation (BERTScore, COMET)
# ─────────────────────────────────────────────────────────────────────────────

def compute_bertscore(items: list, batch_size: int = 32) -> list:
    try:
        from bert_score import score as bscore
        import torch
        hyps = [it.get("flores_passage", "") for it in items]
        refs = [it.get("flores_passage_en", it.get("flores_passage", "")) for it in items]
        logger.info(f"  BERTScore: {len(hyps)} items...")
        P, R, F = bscore(hyps, refs, lang="xx", model_type="microsoft/mdeberta-v3-base",
                          batch_size=batch_size, device="cuda" if torch.cuda.is_available() else "cpu",
                          verbose=False, rescale_with_baseline=False)
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        return F.tolist()
    except ImportError:
        logger.warning("bert_score not installed. Run: pip install bert-score")
        return [None] * len(items)
    except Exception as e:
        logger.error(f"BERTScore failed: {e}")
        return [None] * len(items)

def compute_comet(items: list) -> list:
    try:
        from comet import download_model, load_from_checkpoint
        import torch
        logger.info("  COMET: loading model...")
        comet_path = download_model("Unbabel/wmt22-comet-da")
        model = load_from_checkpoint(comet_path)
        data = [
            {
                "src": it.get("flores_passage_en", ""),
                "mt": it.get("flores_passage", ""),
                "ref": it.get("flores_passage_en", ""),  # using EN as ref
            }
            for it in items
        ]
        logger.info(f"  COMET: scoring {len(data)} items...")
        output = model.predict(data, batch_size=16, gpus=1 if torch.cuda.is_available() else 0)
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        return output.scores
    except ImportError:
        logger.warning("COMET not installed. Run: pip install unbabel-comet")
        return [None] * len(items)
    except Exception as e:
        logger.error(f"COMET failed: {e}")
        return [None] * len(items)


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate stats
# ─────────────────────────────────────────────────────────────────────────────
def aggregate(scores: list, label: str) -> dict:
    valid = [s for s in scores if s is not None]
    if not valid:
        return {f"{label}_mean": None, f"{label}_std": None,
                f"{label}_min": None, f"{label}_max": None}
    mean = sum(valid) / len(valid)
    std  = math.sqrt(sum((x-mean)**2 for x in valid)/len(valid))
    return {
        f"{label}_mean": round(mean, 4),
        f"{label}_std":  round(std, 4),
        f"{label}_min":  round(min(valid), 4),
        f"{label}_max":  round(max(valid), 4),
        f"{label}_n":    len(valid),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Comparison report
# ─────────────────────────────────────────────────────────────────────────────
def generate_comparison_report(before_path: str, after_path: str, out_dir: str):
    with open(before_path) as f: before = json.load(f)
    with open(after_path)  as f: after  = json.load(f)

    metrics = ["chrf", "bleu", "meteor", "labse", "bge_m3", "qwen_embedding",
               "bertscore", "comet"]

    lines = [
        "# Translation Quality: Before vs After Frontier-LLM Correction",
        f"\n**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"\n**Stage 1 (Gemini)**: google/gemini-3.7-flash:batch",
        f"\n**Stage 2 (Qwen)**:   qwen/qwen3-7-flash",
        "\n---\n",
    ]

    for lang in ["ory_Orya", "asm_Beng", "sin_Sinh", "mlt_Latn"]:
        lang_name = LANG_NAMES.get(lang, lang)
        b = before.get("per_language", {}).get(lang, {})
        a = after.get("per_language", {}).get(lang, {})
        if not b or not a: continue

        lines.append(f"\n## {lang_name} (`{lang}`)\n")
        lines.append(f"| Metric | Before | After | Δ | Direction |")
        lines.append(f"|--------|--------|-------|---|-----------|")
        for m in metrics:
            key = f"{m}_mean"
            bv = b.get(key)
            av = a.get(key)
            if bv is None or av is None: continue
            delta = av - bv
            arrow = "✅ ↑" if delta > 0.01 else ("❌ ↓" if delta < -0.01 else "➡ =")
            lines.append(f"| **{m.upper()}** | {bv:.4f} | {av:.4f} | {delta:+.4f} | {arrow} |")

    # Token usage
    token_log = "ikdd/correction_token_usage.jsonl"
    if os.path.exists(token_log):
        lines.append("\n---\n\n## Token Usage (API Cost)\n")
        lines.append("| Stage | Model | Items | Prompt Tokens | Completion Tokens | Total |")
        lines.append("|-------|-------|-------|---------------|-------------------|-------|")
        total_tokens = 0
        with open(token_log) as f:
            for line in f:
                e = json.loads(line.strip())
                lines.append(
                    f"| {e['stage']} | `{e['model']}` | {e['n_items']:,} | "
                    f"{e['prompt_tokens']:,} | {e['completion_tokens']:,} | "
                    f"{e['total_tokens']:,} |"
                )
                total_tokens += e['total_tokens']
        lines.append(f"\n**Total tokens used**: {total_tokens:,}")

    report_path = os.path.join(out_dir, "comparison_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info(f"Comparison report → {report_path}")
    print("\n".join(lines))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", default="translated")
    p.add_argument("--output-dir", default="eval_results")
    p.add_argument("--tag", default="eval", help="Label for output files (e.g. before_correction)")
    p.add_argument("--languages", nargs="+",
                   default=["ory_Orya", "asm_Beng", "sin_Sinh", "mlt_Latn"])
    p.add_argument("--labse-path", default="sentence-transformers/LaBSE")
    p.add_argument("--skip-tier2", action="store_true", help="Skip embedding metrics")
    p.add_argument("--skip-tier3", action="store_true", help="Skip BERTScore and COMET")
    p.add_argument("--skip-bertscore", action="store_true")
    p.add_argument("--skip-comet", action="store_true")
    p.add_argument("--skip-download", action="store_true",
                   help="Don't download models not found locally")
    p.add_argument("--compare", action="store_true", help="Generate before/after comparison")
    p.add_argument("--before", default="eval_results/eval_before_correction_summary.json")
    p.add_argument("--after",  default="eval_results/eval_after_correction_summary.json")
    return p.parse_args()


def main():
    args = parse_args()

    if args.compare:
        generate_comparison_report(args.before, args.after, args.output_dir)
        return

    os.makedirs(args.output_dir, exist_ok=True)
    prefix = os.path.join(args.output_dir, f"eval_{args.tag}")

    logger.info("=" * 70)
    logger.info(f"TRANSLATION QUALITY EVALUATION — tag: {args.tag}")
    logger.info(f"  Input:  {args.input_dir}")
    logger.info(f"  Output: {args.output_dir}")
    logger.info("=" * 70)

    # Detect file prefix (translated_ or corrected_)
    all_items_flat = []
    per_lang = {}

    for lang in args.languages:
        for prefix_name in ["corrected_", "translated_"]:
            f = os.path.join(args.input_dir, f"{prefix_name}{lang}.jsonl")
            if os.path.exists(f):
                items = [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
                per_lang[lang] = items
                all_items_flat.extend(items)
                logger.info(f"  [{lang}] {len(items)} items from {f}")
                break

    if not all_items_flat:
        logger.error(f"No items found in {args.input_dir}!")
        sys.exit(1)

    logger.info(f"Total items: {len(all_items_flat)}")

    # ── Tier 1 ────────────────────────────────────────────────────────────────
    logger.info("\nTier 1: chrF++, BLEU, METEOR (fast)")
    t0 = time.time()
    tier1_all = compute_tier1(all_items_flat)
    logger.info(f"  Done in {time.time()-t0:.1f}s")

    # Build per-item score dict
    item_scores = {(s["item_id"], s["lang_code"]): s for s in tier1_all}

    # ── Tier 2 ────────────────────────────────────────────────────────────────
    tier2_scores = {}
    if not args.skip_tier2:
        logger.info("\nTier 2: Embedding similarities (GPU)")
        t0 = time.time()
        tier2_scores = compute_tier2(all_items_flat, args)
        logger.info(f"  Tier 2 done in {time.time()-t0:.1f}s")
        for metric, scores in tier2_scores.items():
            for item, score in zip(all_items_flat, scores):
                key = (item["item_id"], item["lang_code"])
                item_scores[key][metric] = round(score, 4) if score is not None else None

    # ── Tier 3 ────────────────────────────────────────────────────────────────
    if not args.skip_tier3:
        if not args.skip_bertscore:
            logger.info("\nTier 3a: BERTScore (mDeBERTa)")
            t0 = time.time()
            bs_scores = compute_bertscore(all_items_flat)
            logger.info(f"  BERTScore done in {time.time()-t0:.1f}s")
            for item, score in zip(all_items_flat, bs_scores):
                key = (item["item_id"], item["lang_code"])
                item_scores[key]["bertscore"] = round(score, 4) if score is not None else None

        if not args.skip_comet:
            logger.info("\nTier 3b: COMET-22 (wmt22-comet-da)")
            t0 = time.time()
            comet_scores = compute_comet(all_items_flat)
            logger.info(f"  COMET done in {time.time()-t0:.1f}s")
            for item, score in zip(all_items_flat, comet_scores):
                key = (item["item_id"], item["lang_code"])
                item_scores[key]["comet"] = round(score, 4) if score is not None else None

    # ── Save full results ──────────────────────────────────────────────────────
    results_path = f"{prefix}.json"
    all_scores_list = list(item_scores.values())
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({
            "metadata": {
                "tag": args.tag,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "input_dir": args.input_dir,
                "n_items": len(all_scores_list),
                "languages": args.languages,
                "metrics": ["chrf", "bleu", "meteor"]
                            + list(tier2_scores.keys())
                            + (["bertscore"] if not args.skip_bertscore and not args.skip_tier3 else [])
                            + (["comet"]     if not args.skip_comet     and not args.skip_tier3 else []),
            },
            "items": all_scores_list,
        }, f, indent=2, ensure_ascii=False)
    logger.info(f"\nFull results → {results_path}")

    # ── Per-language aggregate summary ────────────────────────────────────────
    metrics_available = (
        ["chrf", "bleu", "meteor"]
        + list(tier2_scores.keys())
        + (["bertscore"] if not args.skip_bertscore and not args.skip_tier3 else [])
        + (["comet"]     if not args.skip_comet     and not args.skip_tier3 else [])
    )

    per_lang_summary = {}
    for lang in args.languages:
        lang_scores = [s for s in all_scores_list if s["lang_code"] == lang]
        if not lang_scores: continue
        summary = {"n": len(lang_scores)}
        for metric in metrics_available:
            vals = [s[metric] for s in lang_scores if s.get(metric) is not None]
            if vals:
                mean = sum(vals)/len(vals)
                std  = math.sqrt(sum((x-mean)**2 for x in vals)/len(vals))
                summary[f"{metric}_mean"] = round(mean, 4)
                summary[f"{metric}_std"]  = round(std, 4)
                summary[f"{metric}_min"]  = round(min(vals), 4)
                summary[f"{metric}_max"]  = round(max(vals), 4)
        per_lang_summary[lang] = summary

    summary_obj = {
        "tag": args.tag,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "per_language": per_lang_summary,
        "metrics": metrics_available,
    }
    summary_path = f"{prefix}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_obj, f, indent=2, ensure_ascii=False)
    logger.info(f"Summary → {summary_path}")

    # ── Print markdown table ───────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info(f"EVALUATION RESULTS — {args.tag}")
    logger.info("=" * 70)

    for lang in args.languages:
        s = per_lang_summary.get(lang, {})
        lang_name = LANG_NAMES.get(lang, lang)
        logger.info(f"\n  [{lang_name}]  (n={s.get('n',0)})")
        for metric in metrics_available:
            mv = s.get(f"{metric}_mean")
            if mv is not None:
                std = s.get(f"{metric}_std", 0)
                logger.info(f"    {metric:<20} {mv:.4f}  ± {std:.4f}")

    # Write markdown report
    report_lines = [
        f"# Translation Quality Report: {args.tag}",
        f"\n**Evaluated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"\n**Input**: `{args.input_dir}`",
        f"\n**Total items**: {len(all_scores_list)}\n",
        "\n## Per-Language Results\n",
    ]
    for lang in args.languages:
        s = per_lang_summary.get(lang, {})
        lang_name = LANG_NAMES.get(lang, lang)
        report_lines.append(f"\n### {lang_name} (`{lang}`, n={s.get('n',0)})\n")
        report_lines.append("| Metric | Mean | Std | Min | Max |")
        report_lines.append("|--------|------|-----|-----|-----|")
        for metric in metrics_available:
            mv = s.get(f"{metric}_mean")
            if mv is None: continue
            report_lines.append(
                f"| **{metric.upper()}** | {mv:.4f} | "
                f"{s.get(f'{metric}_std', 0):.4f} | "
                f"{s.get(f'{metric}_min', 0):.4f} | "
                f"{s.get(f'{metric}_max', 0):.4f} |"
            )

    report_path = f"{prefix}_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    logger.info(f"Markdown report → {report_path}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
