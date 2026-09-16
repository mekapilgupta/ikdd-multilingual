#!/usr/bin/env python3
"""
01_sample_dataset.py
Stratified sampling of 150 items per reasoning level (L0-L5).
Sources:
  L0, L1      -> curated_eng_Latn.jsonl (real Belebele)
  L2          -> curated_eng_Latn.jsonl (27 real) + synthetic_data_v2.jsonl (top up to 150)
  L3, L4, L5  -> synthetic_data_v2.jsonl
Stratification: evenly spaced by flores_passage length (short/medium/long coverage).
Output: ikdd/sampled_eng_Latn.jsonl
"""
import json, random, argparse, os, sys, logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("sampler")

def load_jsonl(path):
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items

def stratified_sample(items, n, key="flores_passage"):
    """Sample n items evenly spread by key length (short → long)."""
    if len(items) <= n:
        return items[:]
    sorted_items = sorted(items, key=lambda x: len(x.get(key, "")))
    # Pick evenly spaced indices
    step = len(sorted_items) / n
    indices = [int(i * step) for i in range(n)]
    return [sorted_items[i] for i in indices]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--curated-file", default="curated_eng_Latn.jsonl")
    parser.add_argument("--synthetic-file", default="synthetic_data_v2.jsonl")
    parser.add_argument("--output-file", default="sampled_eng_Latn.jsonl")
    parser.add_argument("--per-level", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed)

    curated = load_jsonl(args.curated_file) if os.path.exists(args.curated_file) else []
    synthetic = load_jsonl(args.synthetic_file) if os.path.exists(args.synthetic_file) else []

    logger.info(f"Loaded {len(curated)} curated + {len(synthetic)} synthetic items")

    # Group by level
    curated_by_level = {}
    for item in curated:
        lvl = item.get("reasoning_level", "L0")
        curated_by_level.setdefault(lvl, []).append(item)

    synth_by_level = {}
    for item in synthetic:
        lvl = item.get("reasoning_level", "L2")
        synth_by_level.setdefault(lvl, []).append(item)

    N = args.per_level
    output = []

    for level in ["L0", "L1", "L2", "L3", "L4", "L5"]:
        real_pool = curated_by_level.get(level, [])
        synth_pool = synth_by_level.get(level, [])

        # Build combined pool
        if level in ["L0", "L1"]:
            pool = real_pool
        elif level == "L2":
            # real first, then synthetic top-up
            pool = real_pool + synth_pool
        else:
            pool = synth_pool

        sampled = stratified_sample(pool, N)
        # Assign a clean item_id
        for i, item in enumerate(sampled):
            item["item_id"] = f"{level}_{i+1:03d}"
            item["reasoning_level"] = level
        output.extend(sampled)
        logger.info(f"{level}: pool={len(pool)}, sampled={len(sampled)}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        for item in output:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    logger.info(f"Saved {len(output)} items to {args.output_file}")
    from collections import Counter
    dist = Counter(item["reasoning_level"] for item in output)
    for lvl in ["L0","L1","L2","L3","L4","L5"]:
        logger.info(f"  {lvl}: {dist.get(lvl,0)}")

if __name__ == "__main__":
    main()
