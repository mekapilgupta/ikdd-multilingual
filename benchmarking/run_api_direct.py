#!/usr/bin/env python3
"""
run_api_direct.py
=================
Direct (concurrent) API runner for OpenRouter with robust boxed and structured answer parsing.
- Prompts model to provide strictly \boxed{X} (where X is 1, 2, 3, or 4) without reasoning.
- Suppresses reasoning traces where supported (e.g. Qwen, DeepSeek).
- Robust parser extracts the final boxed answer or structured answer, eliminating false positives from internal thinking tokens.
- Supports resuming from cache and concurrency across workers.
"""

import os, sys, time, json, re, argparse, logging, requests, random
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from threading import Lock

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("ikdd/benchmark_direct.log", mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger("direct_benchmark")

API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

LANG_NAMES = {
    "eng_Latn": "English",
    "ory_Orya": "Odia",
    "asm_Beng": "Assamese",
    "sin_Sinh": "Sinhala",
    "mlt_Latn": "Maltese"
}

LETTER_MAP = {"a": "1", "b": "2", "c": "3", "d": "4"}
WORD_MAP = {
    "one": "1", "two": "2", "three": "3", "four": "4",
    "1st": "1", "2nd": "2", "3rd": "3", "4th": "4",
    "first": "1", "second": "2", "third": "3", "fourth": "4"
}


def build_benchmark_prompt(record: dict, condition: str) -> str:
    q_en = record.get("question_en", record.get("question", ""))
    a1_en = record.get("mc_answer1_en", record.get("mc_answer1", ""))
    a2_en = record.get("mc_answer2_en", record.get("mc_answer2", ""))
    a3_en = record.get("mc_answer3_en", record.get("mc_answer3", ""))
    a4_en = record.get("mc_answer4_en", record.get("mc_answer4", ""))

    q_tgt  = record.get("question", q_en)
    a1_tgt = record.get("mc_answer1", a1_en)
    a2_tgt = record.get("mc_answer2", a2_en)
    a3_tgt = record.get("mc_answer3", a3_en)
    a4_tgt = record.get("mc_answer4", a4_en)

    passage_tgt = record.get("flores_passage", "")
    passage_en  = record.get("flores_passage_en", "")
    triples_en  = record.get("triples_str", "")
    triples_tgt = record.get("triples_str_tgt", triples_en)
    lmp         = record.get("length_matched_passage", passage_tgt[:400])
    lang_name   = LANG_NAMES.get(record.get("lang_code", "eng_Latn"), "the target language")

    if condition == "C1":
        evidence = ""
        q_block  = f"Q: {q_en}"
        c_block  = f"1. {a1_en}\n2. {a2_en}\n3. {a3_en}\n4. {a4_en}"
    elif condition == "C2":
        evidence = f"[{lang_name} Passage]\n{passage_tgt}"
        q_block  = f"Q: {q_tgt}"
        c_block  = f"1. {a1_tgt}\n2. {a2_tgt}\n3. {a3_tgt}\n4. {a4_tgt}"
    elif condition == "C3":
        evidence = f"[English Passage]\n{passage_en}"
        q_block  = f"Q: {q_en}"
        c_block  = f"1. {a1_en}\n2. {a2_en}\n3. {a3_en}\n4. {a4_en}"
    elif condition == "C4":
        evidence = f"[English Facts]\n{triples_en or 'N/A'}"
        q_block  = f"Q: {q_en}"
        c_block  = f"1. {a1_en}\n2. {a2_en}\n3. {a3_en}\n4. {a4_en}"
    elif condition == "C5":
        evidence = f"[{lang_name} Facts]\n{triples_tgt or 'N/A'}"
        q_block  = f"Q: {q_tgt}"
        c_block  = f"1. {a1_tgt}\n2. {a2_tgt}\n3. {a3_tgt}\n4. {a4_tgt}"
    elif condition == "C6":
        evidence = f"[{lang_name} Snippet]\n{lmp}"
        q_block  = f"Q: {q_tgt}"
        c_block  = f"1. {a1_tgt}\n2. {a2_tgt}\n3. {a3_tgt}\n4. {a4_tgt}"
    elif condition == "C7":
        evidence = (
            f"[English Facts]\n{triples_en or 'N/A'}\n\n"
            f"[{lang_name} Facts]\n{triples_tgt or 'N/A'}"
        )
        q_block  = f"Q: {q_tgt}"
        c_block  = f"1. {a1_tgt}\n2. {a2_tgt}\n3. {a3_tgt}\n4. {a4_tgt}"
    else:
        evidence = ""
        q_block  = f"Q: {q_en}"
        c_block  = f"1. {a1_en}\n2. {a2_en}\n3. {a3_en}\n4. {a4_en}"

    evidence_block = f"{evidence}\n\n" if evidence else "[No evidence — answer from memory]\n\n"

    prompt = f"{evidence_block}{q_block}\n\nCHOICES:\n{c_block}\n\nIMPORTANT: Reply ONLY with the final option number inside a box formatted strictly as \\boxed{{X}} where X is 1, 2, 3, or 4 (e.g. \\boxed{{2}}). Do not provide any thinking, reasoning, or other text.\nAnswer:"
    return prompt


def parse_answer(text: str, record: dict = None) -> str:
    if not text:
        return "NA"
    t = text.strip()

    # 1. Boxed matches: \boxed{X}, oxed{X}, oxed{X} (Take LAST match)
    boxed_matches = re.findall(r'(?:\\?boxed|\\x08oxed|\\box)\s*\{\s*([1-4a-dA-D])\s*\}', t)
    if boxed_matches:
        val = boxed_matches[-1].lower()
        return LETTER_MAP.get(val, val)

    # 2. JSON matches: {"answer": "4"} or {"answer": 4} (Take LAST match)
    json_matches = re.findall(r'"answer"\s*:\s*"?\s*([1-4a-dA-D])\s*"?', t, re.IGNORECASE)
    if json_matches:
        val = json_matches[-1].lower()
        return LETTER_MAP.get(val, val)

    # 3. Explicit final answer tags (Take LAST match)
    tag_matches = re.findall(r'(?:final answer|correct answer|the answer is|answer|option|choice)\s*(?:is|:|=|-|\.)*\s*([1-4a-dA-D])', t, re.IGNORECASE)
    if tag_matches:
        val = tag_matches[-1].lower()
        return LETTER_MAP.get(val, val)

    # 4. Pure single digit or single letter
    pure_match = re.match(r'^\s*\(?\s*([1-4a-dA-D])\s*[\.\)]?\s*$', t)
    if pure_match:
        val = pure_match.group(1).lower()
        return LETTER_MAP.get(val, val)

    # 5. Check the very last non-empty line
    lines = [line.strip() for line in t.splitlines() if line.strip()]
    if lines:
        last_line = lines[-1]
        last_line_match = re.match(r'^\s*\(?\s*([1-4a-dA-D])\s*[\.\)]?\s*$', last_line)
        if last_line_match:
            val = last_line_match.group(1).lower()
            return LETTER_MAP.get(val, val)
        last_box = re.findall(r'(?:\\?boxed|\\x08oxed|\\box)\s*\{\s*([1-4a-dA-D])\s*\}', last_line)
        if last_box:
            val = last_box[-1].lower()
            return LETTER_MAP.get(val, val)

    # 6. Word match (one, two, three, four) on short responses (< 100 chars)
    if len(t) < 100:
        for word, digit in WORD_MAP.items():
            if re.search(r'\b' + word + r'\b', t.lower()):
                return digit

    # 7. Option Text Matching on the LAST line or short responses (< 250 chars)
    if record and len(t) < 250:
        check_text = lines[-1].lower() if lines else t.lower()
        for i in [1, 2, 3, 4]:
            for opt_key in [f"mc_answer{i}", f"mc_answer{i}_en"]:
                opt_val = (record.get(opt_key) or "").strip().lower()
                if opt_val and len(opt_val) >= 4 and opt_val in check_text:
                    return str(i)

    return "NA"


def call_api_single(model: str, prompt: str, headers: dict,
                    max_retries: int = 8, base_wait: float = 4.0) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 256,
        "temperature": 0.0,
    }
    
    if any(m in model.lower() for m in ["qwen", "deepseek"]):
        payload["reasoning"] = {"effort": "none"}
    elif "gemini" in model.lower():
        payload["thinking"] = {"budget_tokens": 0}

    for attempt in range(1, max_retries + 1):
        try:
            res = requests.post(BASE_URL, headers=headers, json=payload, timeout=45)
            if res.status_code == 200:
                data = res.json()
                msg = (data.get("choices") or [{}])[0].get("message", {})
                content = msg.get("content")
                if content:
                    return content.strip()
                reasoning = msg.get("reasoning")
                if reasoning:
                    return reasoning.strip()
                return ""
            elif res.status_code == 400 and "reasoning" in payload:
                del payload["reasoning"]
                continue
            elif res.status_code == 429:
                wait = base_wait * (2 ** min(attempt - 1, 3)) + random.uniform(1.0, 3.0)
                logger.warning(f"[429 Rate Limit] Attempt {attempt}/{max_retries}. Backing off {wait:.1f}s...")
                time.sleep(wait)
            elif res.status_code in (500, 502, 503, 504):
                wait = base_wait + random.uniform(1.0, 3.0)
                logger.warning(f"[{res.status_code} Server Error] Attempt {attempt}. Retrying in {wait:.1f}s...")
                time.sleep(wait)
            else:
                logger.error(f"[HTTP {res.status_code}] Response: {res.text[:200]}")
                time.sleep(3.0)
        except Exception as ex:
            logger.warning(f"[Network/Timeout Error] Attempt {attempt}: {ex}")
            time.sleep(base_wait)

    logger.error("All retries exhausted for prompt.")
    return ""


def process_one(task: dict, headers: dict, model: str) -> dict:
    item = task["item"]
    cond = task["cond"]
    prompt = task["prompt"]

    raw_text = call_api_single(model, prompt, headers)
    pred_ans = parse_answer(raw_text, item)
    
    gold_ans = str(item.get("correct_answer_num", "")).strip()
    is_corr = (pred_ans == gold_ans)

    return {
        "item_id": item["item_id"],
        "lang_code": item.get("lang_code", task["lang"]),
        "condition": cond,
        "reasoning_level": item.get("reasoning_level", ""),
        "error_category": item.get("error_category", ""),
        "gold_answer": gold_ans,
        "predicted_answer": pred_ans,
        "is_correct": is_corr,
        "raw_response": raw_text[:300],
    }


def print_summary(out_file: str, conditions: list):
    records = []
    if not os.path.exists(out_file):
        return
    with open(out_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass
    total = len(records)
    by_lang = defaultdict(lambda: {"total": 0, "correct": 0, "na": 0})
    by_cond = defaultdict(lambda: {"total": 0, "correct": 0, "na": 0})

    for r in records:
        lang = r.get("lang_code", "unknown")
        cond = r.get("condition", "unknown")
        corr = 1 if r.get("is_correct") else 0
        na   = 1 if r.get("predicted_answer") == "NA" else 0

        by_lang[lang]["total"] += 1
        by_lang[lang]["correct"] += corr
        by_lang[lang]["na"] += na

        by_cond[cond]["total"] += 1
        by_cond[cond]["correct"] += corr
        by_cond[cond]["na"] += na

    total_corr = sum(r.get("is_correct", False) for r in records)
    total_na   = sum(1 for r in records if r.get("predicted_answer") == "NA")

    print("\n" + "=" * 70)
    print(f"BENCHMARK EVALUATION SUMMARY — {out_file}")
    print("=" * 70)
    acc = (total_corr / total * 100) if total else 0
    print(f"Total Evaluated : {total}")
    print(f"Overall Accuracy: {acc:.2f}% ({total_corr}/{total})")
    print(f"Total NA / Parse: {total_na} ({(total_na/total*100) if total else 0:.2f}%)")
    print("-" * 70)
    print("ACCURACY BY CONDITION:")
    for cond in conditions:
        if cond in by_cond and by_cond[cond]["total"] > 0:
            c_tot = by_cond[cond]["total"]
            c_cor = by_cond[cond]["correct"]
            c_na  = by_cond[cond]["na"]
            c_acc = (c_cor / c_tot) * 100
            print(f"  {cond:4s}: {c_acc:6.2f}% ({c_cor:5d}/{c_tot:5d}) | NA: {c_na:3d}")
    print("-" * 70)
    print("ACCURACY BY LANGUAGE:")
    for lang, st in sorted(by_lang.items()):
        if st["total"] > 0:
            l_acc = (st["correct"] / st["total"]) * 100
            print(f"  {lang:12s}: {l_acc:6.2f}% ({st['correct']:5d}/{st['total']:5d}) | NA: {st['na']:3d}")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model name on OpenRouter")
    parser.add_argument("--data-dir", default="ikdd/corrected")
    parser.add_argument("--output-dir", default="ikdd/benchmarking/results")
    parser.add_argument("--languages", nargs="+", default=["ory_Orya", "asm_Beng", "sin_Sinh", "mlt_Latn"])
    parser.add_argument("--conditions", nargs="+", default=["C1","C2","C3","C4","C5","C6","C7"])
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    model_slug = args.model.replace("/", "_").replace(":", "_").replace(".", "-")
    out_file = os.path.join(args.output_dir, f"benchmark_{model_slug}_direct.jsonl")

    logger.info("=" * 70)
    logger.info("MULTILINGUAL BENCHMARK — DIRECT CONCURRENT EVALUATION")
    logger.info(f"  Model      : {args.model}")
    logger.info(f"  Conditions : {args.conditions}")
    logger.info(f"  Output     : {out_file}")
    logger.info(f"  Workers    : {args.workers}")
    logger.info("=" * 70)

    # 1. Load cache - skip only rows that got a valid 1/2/3/4 prediction
    cached_keys = set()
    if os.path.exists(out_file):
        with open(out_file, "r", encoding="utf-8") as fc:
            for line in fc:
                if line.strip():
                    try:
                        d = json.loads(line)
                        if d.get("predicted_answer") in ("1", "2", "3", "4"):
                            cached_keys.add((d["item_id"], d["lang_code"], d["condition"]))
                    except Exception:
                        pass
    logger.info(f"Loaded {len(cached_keys)} valid cached predictions.")

    # 2. Build task list
    all_tasks = []
    for lang in args.languages:
        fpath = os.path.join(args.data_dir, f"corrected_{lang}.jsonl")
        if not os.path.exists(fpath):
            logger.warning(f"File not found: {fpath}")
            continue
        items = [json.loads(l) for l in open(fpath, encoding="utf-8") if l.strip()]
        for item in items:
            for cond in args.conditions:
                key = (item["item_id"], lang, cond)
                if key not in cached_keys:
                    all_tasks.append({
                        "item": item,
                        "lang": lang,
                        "cond": cond,
                        "prompt": build_benchmark_prompt(item, cond),
                    })

    total_target = len(args.languages) * 900 * len(args.conditions)
    logger.info(f"Total target : {total_target}")
    logger.info(f"Remaining    : {len(all_tasks)}")

    if not all_tasks:
        logger.info("All evaluations completed! Printing summary...")
        print_summary(out_file, args.conditions)
        return

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "X-Title": "Multilingual-Benchmark-Direct",
    }

    file_lock = Lock()
    done_count = [len(cached_keys)]
    na_count = [0]

    def write_result(rec: dict):
        with file_lock:
            with open(out_file, "a", encoding="utf-8") as fa:
                fa.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fa.flush()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_one, task, headers, args.model): task for task in all_tasks}
        for future in as_completed(futures):
            try:
                res = future.result()
                write_result(res)
                done_count[0] += 1
                if res["predicted_answer"] == "NA":
                    na_count[0] += 1
                if done_count[0] % args.progress_every == 0 or done_count[0] == total_target:
                    pct = (done_count[0] / total_target) * 100
                    logger.info(f"Progress: {done_count[0]}/{total_target} ({pct:.1f}%) | NA count: {na_count[0]}")
            except Exception as ex:
                logger.error(f"Task failure: {ex}")

    logger.info("=" * 70)
    logger.info(f"BENCHMARK COMPLETE! Saved to {out_file}")
    logger.info("=" * 70)
    print_summary(out_file, args.conditions)


if __name__ == "__main__":
    main()
