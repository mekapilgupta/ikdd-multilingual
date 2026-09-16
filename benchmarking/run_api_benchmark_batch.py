#!/usr/bin/env python3
"""
run_api_benchmark_batch.py  [LOGIN / CPU NODE]
================================================
Evaluates paid models via OpenRouter Batch API across all 3,600 items x 6 conditions (C1-C6):
  1. google/gemini-3.7-flash:batch
  2. openai/gpt-5.6-luna:batch

Features:
  - Controlled batch concurrency (max_active_batches=2) to strictly respect
    provider enqueued token limits (e.g. OpenAI 1M enqueued token limit).
  - Automatic retry and re-queueing if a batch fails or is cancelled.
  - Rate-limit backoff (HTTP 429) on batch submissions.
  - Resume from existing cache: skips any already-evaluated (item_id, lang, condition) pairs.
  - Line-by-line streaming flush to disk.
  - Comprehensive accuracy evaluation report upon completion.
"""

import os, sys, time, json, re, argparse, logging, requests
from datetime import datetime, timezone
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("ikdd/benchmark_api.log", mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger("api_benchmark")

API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

LANG_NAMES = {
    "eng_Latn": "English", "ory_Orya": "Odia",
    "asm_Beng": "Assamese", "sin_Sinh": "Sinhala", "mlt_Latn": "Maltese"
}
CONDITIONS = ["C1", "C2", "C3", "C4", "C5", "C6"]


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
        evidence = "[NO EVIDENCE PROVIDED — answer from memory only]"
        q_block  = f"QUESTION:\n{q_en}"
        c_block  = f"1. {a1_en}\n2. {a2_en}\n3. {a3_en}\n4. {a4_en}"
    elif condition == "C2":
        evidence = f"PASSAGE ({lang_name}):\n{passage_tgt}"
        q_block  = f"QUESTION ({lang_name}):\n{q_tgt}"
        c_block  = f"1. {a1_tgt}\n2. {a2_tgt}\n3. {a3_tgt}\n4. {a4_tgt}"
    elif condition == "C3":
        evidence = f"PASSAGE (English translation):\n{passage_en}"
        q_block  = f"QUESTION (English):\n{q_en}"
        c_block  = f"1. {a1_en}\n2. {a2_en}\n3. {a3_en}\n4. {a4_en}"
    elif condition == "C4":
        evidence = f"FACT TRIPLES (English):\n{triples_en or '[No triples available]'}"
        q_block  = f"QUESTION (English):\n{q_en}"
        c_block  = f"1. {a1_en}\n2. {a2_en}\n3. {a3_en}\n4. {a4_en}"
    elif condition == "C5":
        evidence = f"FACT TRIPLES ({lang_name}):\n{triples_tgt or '[No triples available]'}"
        q_block  = f"QUESTION ({lang_name}):\n{q_tgt}"
        c_block  = f"1. {a1_tgt}\n2. {a2_tgt}\n3. {a3_tgt}\n4. {a4_tgt}"
    elif condition == "C6":
        evidence = f"PASSAGE SNIPPET ({lang_name}):\n{lmp}"
        q_block  = f"QUESTION ({lang_name}):\n{q_tgt}"
        c_block  = f"1. {a1_tgt}\n2. {a2_tgt}\n3. {a3_tgt}\n4. {a4_tgt}"
    else:
        evidence = ""
        q_block  = f"QUESTION:\n{q_en}"
        c_block  = f"1. {a1_en}\n2. {a2_en}\n3. {a3_en}\n4. {a4_en}"

    prompt = f"""You are an expert reading comprehension evaluator.
{evidence}

{q_block}

CHOICES:
{c_block}

Respond ONLY with a raw JSON object containing two keys:
1. "answer": exact option number ("1", "2", "3", or "4")
2. "reasoning": a concise 10-20 word sentence explaining why you chose this option.

Example JSON output:
{{"answer": "2", "reasoning": "The passage explicitly states that eBay acquired Skype to diversify its profit sources."}}
"""
    return prompt


def parse_benchmark_response(text: str, record: dict) -> dict:
    text = (text or "").strip()
    if not text:
        return {"answer": "NA", "reasoning": "Empty response"}

    # 1. Try parsing JSON
    try:
        start = text.find("{")
        end   = text.rfind("}")
        if start != -1 and end > start:
            data = json.loads(text[start:end+1])
            ans = str(data.get("answer", "")).strip()
            reason = str(data.get("reasoning", "")).strip()
            if ans in {"1", "2", "3", "4"}:
                return {"answer": ans, "reasoning": reason or "Parsed JSON answer"}
    except Exception:
        pass

    # 2. Match patterns like "answer": "1" or "answer": 1
    m_ans = re.search(r'["\']?answer["\']?\s*:\s*["\']?([1-4])["\']?', text, re.IGNORECASE)
    if m_ans:
        m_rea = re.search(r'["\']?reasoning["\']?\s*:\s*["\']([^"\']+)["\']', text, re.IGNORECASE)
        reason = m_rea.group(1).strip() if m_rea else text[:100]
        return {"answer": m_ans.group(1), "reasoning": reason}

    # 3. Match leading or standalone digits
    m_dig = re.search(r'\b([1-4])\b', text)
    if m_dig:
        return {"answer": m_dig.group(1), "reasoning": text[:100]}

    # 4. Match option text if model repeated option
    t_low = text.lower()
    for num in [1, 2, 3, 4]:
        opt_en = (record.get(f"mc_answer{num}_en") or record.get(f"mc_answer{num}") or "").strip().lower()
        opt_tgt = (record.get(f"mc_answer{num}") or "").strip().lower()
        if (opt_en and len(opt_en) > 1 and opt_en in t_low) or (opt_tgt and len(opt_tgt) > 1 and opt_tgt in t_low):
            return {"answer": str(num), "reasoning": f"Matched option text '{opt_en or opt_tgt}'"}

    return {"answer": "NA", "reasoning": "Parse failed"}


def submit_batch_with_retry(payload: dict, headers: dict, max_retries: int = 10) -> str:
    """Submits a batch request to OpenRouter with backoff on HTTP 429 or connection errors."""
    url = "https://openrouter.ai/api/beta/batches"
    for attempt in range(1, max_retries + 1):
        try:
            res = requests.post(url, headers=headers, json=payload, timeout=60)
            if res.status_code in (200, 202):
                data = res.json()
                bid = data.get("id")
                if bid:
                    return bid
                logger.error(f"Submit returned {res.status_code} but no batch ID in response: {res.text[:200]}")
            elif res.status_code == 429:
                wait_time = 15 * attempt
                logger.warning(f"Rate limit HTTP 429 on submit attempt {attempt}/{max_retries}. Sleeping {wait_time}s...")
                time.sleep(wait_time)
            else:
                logger.warning(f"Submit attempt {attempt}/{max_retries} returned HTTP {res.status_code}: {res.text[:200]}. Waiting 10s...")
                time.sleep(10)
        except Exception as ex:
            logger.warning(f"Submit attempt {attempt}/{max_retries} exception: {ex}. Waiting 10s...")
            time.sleep(10)
    return ""


def print_summary(out_file: str):
    """Prints a summary breakdown of all benchmark results in out_file."""
    if not os.path.exists(out_file):
        return

    records = []
    with open(out_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass

    total = len(records)
    if total == 0:
        logger.info(f"No records found in {out_file}")
        return

    logger.info("=" * 70)
    logger.info(f"BENCHMARK SUMMARY ({total} rows in {out_file})")
    logger.info("=" * 70)

    by_lang = defaultdict(lambda: {"total": 0, "correct": 0})
    by_cond = defaultdict(lambda: {"total": 0, "correct": 0})
    by_lang_cond = defaultdict(lambda: {"total": 0, "correct": 0})

    for r in records:
        lang = r.get("lang_code", "unknown")
        cond = r.get("condition", "unknown")
        corr = 1 if r.get("is_correct") else 0

        by_lang[lang]["total"] += 1
        by_lang[lang]["correct"] += corr

        by_cond[cond]["total"] += 1
        by_cond[cond]["correct"] += corr

        by_lang_cond[(lang, cond)]["total"] += 1
        by_lang_cond[(lang, cond)]["correct"] += corr

    logger.info("--- By Language ---")
    for lang in sorted(by_lang.keys()):
        cnt = by_lang[lang]["total"]
        acc = (by_lang[lang]["correct"] / cnt * 100) if cnt else 0
        logger.info(f"  {lang:12s}: {by_lang[lang]['correct']}/{cnt} ({acc:.2f}%)")

    logger.info("--- By Condition (C1 to C6) ---")
    for cond in CONDITIONS:
        if cond in by_cond:
            cnt = by_cond[cond]["total"]
            acc = (by_cond[cond]["correct"] / cnt * 100) if cnt else 0
            logger.info(f"  {cond:4s}: {by_cond[cond]['correct']}/{cnt} ({acc:.2f}%)")

    logger.info("=" * 70)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemini-3.7-flash:batch")
    parser.add_argument("--data-dir", default="ikdd/corrected")
    parser.add_argument("--output-dir", default="ikdd/benchmarking/results")
    parser.add_argument("--languages", nargs="+", default=["ory_Orya", "asm_Beng", "sin_Sinh", "mlt_Latn"])
    parser.add_argument("--batch-size", type=int, default=500, help="Batch size per sub-batch (500 avoids enqueued token limits)")
    parser.add_argument("--max-active-batches", type=int, default=2, help="Max concurrent active batches to stay well within provider enqueued limits")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    model_slug_clean = args.model.replace("/", "_").replace(":", "_")
    out_file = os.path.join(args.output_dir, f"benchmark_{model_slug_clean}.jsonl")

    logger.info("=" * 70)
    logger.info("MULTILINGUAL BENCHMARK — BATCH API RUNNER")
    logger.info(f"  Model              : {args.model}")
    logger.info(f"  Data Directory     : {args.data_dir}")
    logger.info(f"  Output File        : {out_file}")
    logger.info(f"  Batch Size         : {args.batch_size}")
    logger.info(f"  Max Active Batches : {args.max_active_batches}")
    logger.info("=" * 70)

    # 1. Load existing cache from output file
    cached_keys = set()
    if os.path.exists(out_file):
        with open(out_file, "r", encoding="utf-8") as fcache:
            for line in fcache:
                line = line.strip()
                if not line:
                    continue
                try:
                    cdata = json.loads(line)
                    key = (cdata.get("item_id"), cdata.get("lang_code"), cdata.get("condition"))
                    if cdata.get("predicted_answer") != "NA":
                        cached_keys.add(key)
                except Exception:
                    pass

    # 2. Collect all evaluation pairs
    all_prompts = []
    item_map = {}
    for lang in args.languages:
        fpath = os.path.join(args.data_dir, f"corrected_{lang}.jsonl")
        if not os.path.exists(fpath):
            logger.warning(f"File not found: {fpath}, skipping...")
            continue
        items = [json.loads(l) for l in open(fpath, encoding="utf-8") if l.strip()]
        for item in items:
            item_map[item["item_id"]] = item
            for cond in CONDITIONS:
                prompt = build_benchmark_prompt(item, cond)
                all_prompts.append((item["item_id"], lang, cond, item, prompt))

    total_expected = len(all_prompts)
    prompts_to_submit = [p for p in all_prompts if (p[0], p[1], p[2]) not in cached_keys]

    logger.info(f"Total Target Prompts: {total_expected}")
    logger.info(f"Already Cached      : {len(cached_keys)}")
    logger.info(f"Remaining To Run    : {len(prompts_to_submit)}")

    if not prompts_to_submit:
        logger.info("All benchmark prompts are already cached and complete!")
        print_summary(out_file)
        return

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    # Split prompts_to_submit into chunks of args.batch_size
    prompt_chunks = []
    for i in range(0, len(prompts_to_submit), args.batch_size):
        prompt_chunks.append(prompts_to_submit[i:i+args.batch_size])

    logger.info(f"Created {len(prompt_chunks)} sub-batches of size <= {args.batch_size}.")

    active_batches = {}  # batch_id -> {"chunk": chunk, "submitted_at": timestamp}
    chunk_index = 0

    # 3. Dynamic Sliding Batch Execution Loop
    while chunk_index < len(prompt_chunks) or active_batches:
        # Fill active batch queue up to max_active_batches
        while len(active_batches) < args.max_active_batches and chunk_index < len(prompt_chunks):
            chunk = prompt_chunks[chunk_index]
            chunk_num = chunk_index + 1
            chunk_index += 1

            requests_payload = []
            for (iid, lang, cond, item, prompt) in chunk:
                requests_payload.append({
                    "custom_id": f"{iid}|{lang}|{cond}",
                    "body": {
                        "model": args.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 256,
                        "temperature": 0.0,
                    }
                })

            payload = {
                "endpoint": "/v1/chat/completions",
                "model": args.model,
                "requests": requests_payload,
            }

            logger.info(f"Submitting Batch {chunk_num}/{len(prompt_chunks)} ({len(requests_payload)} requests)...")
            batch_id = submit_batch_with_retry(payload, headers)
            if not batch_id:
                logger.error(f"Failed to submit Batch {chunk_num} after retries. Re-queueing chunk...")
                prompt_chunks.append(chunk)
                continue

            logger.info(f"Batch {chunk_num} submitted successfully! ID: {batch_id}")
            active_batches[batch_id] = {
                "chunk": chunk,
                "chunk_num": chunk_num,
                "submitted_at": time.time(),
                "total_reqs": len(requests_payload),
            }
            time.sleep(2)  # Short pause between batch submissions to be gentle on API

        # Poll all active batches
        time.sleep(10)
        finished_batch_ids = []

        for batch_id, meta in list(active_batches.items()):
            poll_url = f"https://openrouter.ai/api/beta/batches/{batch_id}"
            try:
                poll_res = requests.get(poll_url, headers=headers, timeout=30)
                if poll_res.status_code != 200:
                    logger.warning(f"Poll for {batch_id[-12:]} returned HTTP {poll_res.status_code}")
                    continue

                data = poll_res.json()
                status = data.get("status")
                req_counts = data.get("request_counts", {})
                completed = req_counts.get("completed", 0)
                total = req_counts.get("total", meta["total_reqs"])

                elapsed_min = (time.time() - meta["submitted_at"]) / 60.0
                logger.info(f"Batch {batch_id[-12:]} (Batch #{meta['chunk_num']}): status={status} ({completed}/{total} completed, {elapsed_min:.1f}m elapsed)")

                if status == "completed":
                    batch_results = data.get("results") or []
                    logger.info(f"Unpacking & saving {len(batch_results)} results for batch {batch_id[-12:]}...")

                    # Write results to output file
                    with open(out_file, "a", encoding="utf-8") as fa:
                        for item_res in batch_results:
                            if not isinstance(item_res, dict):
                                continue
                            cid = item_res.get("custom_id", "")
                            if cid.count("|") < 2:
                                continue
                            item_id, lang, cond = cid.split("|", 2)
                            key = (item_id, lang, cond)
                            if key in cached_keys:
                                continue

                            item = item_map.get(item_id, {})
                            resp = item_res.get("response") or {}
                            resp_body = resp.get("body") or {}
                            text = ""
                            try:
                                text = resp_body["choices"][0]["message"]["content"] or ""
                            except Exception:
                                text = ""

                            parsed = parse_benchmark_response(text, item)
                            gold_ans = str(item.get("correct_answer_num", "")).strip()
                            is_correct = (parsed["answer"] == gold_ans)

                            rec_out = {
                                "item_id": item_id,
                                "lang_code": lang,
                                "condition": cond,
                                "reasoning_level": item.get("reasoning_level", ""),
                                "error_category": item.get("error_category", ""),
                                "gold_answer": gold_ans,
                                "predicted_answer": parsed["answer"],
                                "is_correct": is_correct,
                                "reasoning": parsed["reasoning"],
                                "raw_response": (text or "")[:300],
                            }
                            fa.write(json.dumps(rec_out, ensure_ascii=False) + "\n")
                            fa.flush()
                            cached_keys.add(key)

                    finished_batch_ids.append(batch_id)
                    logger.info(f"Overall Progress: {len(cached_keys)} / {total_expected} ({len(cached_keys)/total_expected*100:.1f}%) cached.")

                elif status in ["failed", "cancelled", "expired"]:
                    err_msg = data.get("error") or data.get("message") or "Unknown error"
                    logger.error(f"Batch {batch_id[-12:]} ended with status='{status}': {err_msg}")
                    # Re-queue missing items from this chunk
                    uncompleted = [p for p in meta["chunk"] if (p[0], p[1], p[2]) not in cached_keys]
                    if uncompleted:
                        logger.warning(f"Re-queueing {len(uncompleted)} uncompleted items from batch {batch_id[-12:]}...")
                        prompt_chunks.append(uncompleted)
                    finished_batch_ids.append(batch_id)

            except Exception as ex:
                logger.warning(f"Error polling batch {batch_id[-12:]}: {ex}")

        for bid in finished_batch_ids:
            active_batches.pop(bid, None)

    logger.info("=" * 70)
    logger.info(f"ALL BENCHMARK BATCHES COMPLETED! Results saved to {out_file}")
    logger.info("=" * 70)
    print_summary(out_file)


if __name__ == "__main__":
    main()
