#!/usr/bin/env python3
"""
judge_reasons.py — LLM Judge for Benchmark Reason Quality
==========================================================
Evaluates the quality of model-generated reasons for MCQ answers.
Works in two modes:
  --mode vllm   : Uses local Qwen3.5-35B on vLLM
  --mode gemini : Uses Gemini API (more accessible, no GPU needed)
  --mode openai : Uses GPT-4o API

JUDGE EVALUATES per reason:
  1. reason_faithful  : Does the reason cite something actually in the passage?
  2. reason_correct   : Does the reason correctly explain the right answer?
  3. reason_concise   : Is it ≤10 words?
  4. reason_quality   : "good" / "partial" / "wrong"

OUTPUT adds judge fields to each benchmark result record.
"""

import os, gc, json, sys, time, logging, argparse
from typing import List, Optional
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("judge_reasons")

# ─────────────────────────────────────────────────────────────────────────────
# Judge Prompt
# ─────────────────────────────────────────────────────────────────────────────

def build_judge_prompt(record: dict) -> tuple:
    passage_snippet = record.get("passage_en_snippet", "")
    question = record.get("question_en", "")
    mc_answers = record.get("mc_answers_en", ["", "", "", ""])
    correct_ans = record.get("correct_answer", "")
    predicted_ans = record.get("predicted_answer", "")
    reason_en = record.get("reason_en", "")
    is_correct = record.get("is_correct", False)

    if len(mc_answers) < 4:
        mc_answers = (mc_answers + ["", "", "", ""])[:4]
    correct_text = mc_answers[int(correct_ans) - 1] if correct_ans.isdigit() else ""
    predicted_text = mc_answers[int(predicted_ans) - 1] if predicted_ans.isdigit() else "NA"

    system_msg = (
        "You are a strict reading comprehension evaluation judge. "
        "Evaluate the quality of a student's stated reason for their MCQ answer. "
        "Base your evaluation ONLY on the passage snippet provided."
    )
    user_msg = (
        f"PASSAGE SNIPPET (English):\n{passage_snippet}\n\n"
        f"QUESTION:\n{question}\n\n"
        f"CHOICES:\n"
        f"  1. {mc_answers[0]}\n"
        f"  2. {mc_answers[1]}\n"
        f"  3. {mc_answers[2]}\n"
        f"  4. {mc_answers[3]}\n\n"
        f"CORRECT ANSWER: {correct_ans}. \"{correct_text}\"\n"
        f"MODEL'S ANSWER: {predicted_ans}. \"{predicted_text}\" (Answer is {'CORRECT' if is_correct else 'WRONG'})\n"
        f"MODEL'S REASON (English): \"{reason_en}\"\n\n"
        "EVALUATE the model's reason on three criteria:\n"
        "1. faithful: Is the reason based on something actually in the passage? "
        "   (not hallucinated or from outside knowledge)\n"
        "2. correct: Does the reason correctly justify why the CORRECT answer is right? "
        "   (even if model picked wrong answer, reason could still explain the correct one)\n"
        "3. concise: Is the reason 10 words or fewer?\n"
        "4. quality: Overall — 'good' (all 3 pass), 'partial' (1-2 pass), 'wrong' (fails all)\n\n"
        "Output ONLY valid JSON:\n"
        '{"faithful": true/false, "correct": true/false, "concise": true/false, '
        '"quality": "good"/"partial"/"wrong", "judge_note": "one-sentence explanation"}'
    )
    return system_msg, user_msg

def parse_judge_response(text: str) -> Optional[dict]:
    try:
        cleaned = text.strip()
        if "<think>" in cleaned:
            think_end = cleaned.rfind("</think>")
            if think_end != -1:
                cleaned = cleaned[think_end + len("</think>"):].strip()
        if "```json" in cleaned:
            cleaned = cleaned[cleaned.find("```json") + 7:]
            cleaned = cleaned[:cleaned.find("```")]
        elif "```" in cleaned:
            cleaned = cleaned[cleaned.find("```") + 3:]
            cleaned = cleaned[:cleaned.find("```")]
        start = cleaned.find("{")
        if start != -1:
            cleaned = cleaned[start:]
        data = json.loads(cleaned)
        if data.get("quality") not in {"good", "partial", "wrong"}:
            return None
        return {
            "judge_faithful": bool(data.get("faithful", False)),
            "judge_correct": bool(data.get("correct", False)),
            "judge_concise": bool(data.get("concise", True)),
            "judge_quality": str(data.get("quality", "wrong")),
            "judge_note": str(data.get("judge_note", ""))[:300],
        }
    except Exception:
        return None

# ─────────────────────────────────────────────────────────────────────────────
# vLLM Judge
# ─────────────────────────────────────────────────────────────────────────────

def judge_vllm(model_path: str, records: List[dict], batch_size: int = 16,
               max_tokens: int = 512) -> List[Optional[dict]]:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    import torch

    logger.info(f"Loading vLLM judge: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True,
                                               local_files_only=True)
    llm = LLM(model=model_path, tokenizer=model_path, trust_remote_code=True,
               max_model_len=8192, gpu_memory_utilization=0.95, dtype="bfloat16",
               additional_config={"gdn_prefill_backend": "triton"})
    params = SamplingParams(temperature=0.1, max_tokens=max_tokens, top_p=0.9)

    results = [None] * len(records)
    for batch_start in range(0, len(records), batch_size):
        batch = records[batch_start: batch_start + batch_size]
        prompts = []
        for rec in batch:
            sys_msg, user_msg = build_judge_prompt(rec)
            messages = [{"role": "system", "content": sys_msg},
                        {"role": "user", "content": user_msg}]
            try:
                prompt = tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False,
                    enable_thinking=True)
            except TypeError:
                try:
                    prompt = tokenizer.apply_chat_template(
                        messages, add_generation_prompt=True, tokenize=False)
                except:
                    prompt = f"<|im_start|>system\n{sys_msg}<|im_end|>\n<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n<think>\n"
            except:
                prompt = f"<|im_start|>system\n{sys_msg}<|im_end|>\n<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n<think>\n"
            prompts.append({"prompt": prompt})

        outputs = llm.generate(prompts, params)
        for i, out in enumerate(outputs):
            raw = out.outputs[0].text
            results[batch_start + i] = parse_judge_response(raw)
            if (batch_start + i + 1) % 100 == 0:
                logger.info(f"  Judge processed {batch_start + i + 1}/{len(records)}")

    del llm, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return results

# ─────────────────────────────────────────────────────────────────────────────
# Gemini Judge (API — no GPU needed)
# ─────────────────────────────────────────────────────────────────────────────

def judge_gemini(model_name: str, records: List[dict], api_key: str,
                  max_tokens: int = 512, delay_sec: float = 0.3) -> List[Optional[dict]]:
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name,
        generation_config=genai.GenerationConfig(
            temperature=0.0, max_output_tokens=max_tokens,
            response_mime_type="application/json",
        )
    )
    results = []
    for i, rec in enumerate(records):
        sys_msg, user_msg = build_judge_prompt(rec)
        try:
            resp = model.generate_content(f"{sys_msg}\n\n{user_msg}")
            results.append(parse_judge_response(resp.text))
        except Exception as e:
            logger.warning(f"Gemini judge error on item {i}: {e}")
            results.append(None)
        if (i + 1) % 50 == 0:
            logger.info(f"  Gemini judge processed {i + 1}/{len(records)}")
        time.sleep(delay_sec)
    return results

# ─────────────────────────────────────────────────────────────────────────────
# OpenAI Judge (API)
# ─────────────────────────────────────────────────────────────────────────────

def judge_openai(model_name: str, records: List[dict], api_key: str,
                  base_url: Optional[str] = None, max_tokens: int = 512,
                  delay_sec: float = 0.5) -> List[Optional[dict]]:
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url=base_url)
    results = []
    for i, rec in enumerate(records):
        sys_msg, user_msg = build_judge_prompt(rec)
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "system", "content": sys_msg},
                          {"role": "user", "content": user_msg}],
                max_tokens=max_tokens,
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            results.append(parse_judge_response(resp.choices[0].message.content))
        except Exception as e:
            logger.warning(f"OpenAI judge error on item {i}: {e}")
            results.append(None)
        if (i + 1) % 50 == 0:
            logger.info(f"  OpenAI judge processed {i + 1}/{len(records)}")
        time.sleep(delay_sec)
    return results

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LLM Judge for benchmark reason quality.")
    parser.add_argument("--mode", choices=["vllm", "gemini", "openai"], default="vllm")
    parser.add_argument("--input-file", required=True, help="Benchmark results JSONL")
    parser.add_argument("--output-file", required=True, help="Judged results JSONL")
    parser.add_argument("--model-path",
                        default="Qwen/Qwen2.5-32B-Instruct",
                        help="Local model path (vllm mode)")
    parser.add_argument("--model-name", default=None, help="API model name")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--api-delay", type=float, default=0.3)
    parser.add_argument("--skip-already-judged", action="store_true", default=True)
    args = parser.parse_args()

    # Load benchmark results
    records = []
    with open(args.input_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    logger.info(f"Loaded {len(records)} benchmark results.")

    # Resume: skip already judged
    judged_ids = set()
    if args.skip_already_judged and os.path.exists(args.output_file):
        with open(args.output_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        judged_ids.add(json.loads(line)["item_id"])
                    except: pass
        logger.info(f"Resume: {len(judged_ids)} already judged.")

    pending = [r for r in records if r.get("item_id", "") not in judged_ids]
    logger.info(f"Pending: {len(pending)}")

    if not pending:
        logger.info("All records already judged.")
        return

    # Resolve API keys
    if args.mode == "gemini" and not args.api_key:
        args.api_key = os.environ.get("GOOGLE_API_KEY", "")
    elif args.mode == "openai" and not args.api_key:
        args.api_key = os.environ.get("OPENAI_API_KEY", "")

    # Run judge
    if args.mode == "vllm":
        judgments = judge_vllm(args.model_path, pending, batch_size=args.batch_size)
    elif args.mode == "gemini":
        judgments = judge_gemini(args.model_name or "gemini-2.0-flash",
                                  pending, args.api_key, delay_sec=args.api_delay)
    elif args.mode == "openai":
        judgments = judge_openai(args.model_name or "gpt-4o",
                                  pending, args.api_key, args.base_url,
                                  delay_sec=args.api_delay)

    # Write output (original record + judge fields merged)
    with open(args.output_file, "a", encoding="utf-8") as fout:
        for rec, jud in zip(pending, judgments):
            out = dict(rec)
            if jud:
                out.update(jud)
            else:
                out.update({"judge_faithful": None, "judge_correct": None,
                             "judge_concise": None, "judge_quality": "unknown",
                             "judge_note": "parse_failed"})
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            fout.flush()

    # Summary stats
    judged = []
    with open(args.output_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                judged.append(json.loads(line))

    quality_counts = defaultdict(int)
    faithful_count = sum(1 for r in judged if r.get("judge_faithful"))
    correct_count = sum(1 for r in judged if r.get("judge_correct"))
    for r in judged:
        quality_counts[r.get("judge_quality", "unknown")] += 1

    total = len(judged)
    logger.info("=" * 60)
    logger.info("JUDGE SUMMARY")
    logger.info(f"  Total judged:        {total}")
    logger.info(f"  Faithful reasons:    {faithful_count}/{total} = {100*faithful_count/max(total,1):.1f}%")
    logger.info(f"  Correct explanations:{correct_count}/{total} = {100*correct_count/max(total,1):.1f}%")
    logger.info(f"  Quality - good:      {quality_counts['good']}")
    logger.info(f"  Quality - partial:   {quality_counts['partial']}")
    logger.info(f"  Quality - wrong:     {quality_counts['wrong']}")
    logger.info(f"  Output: {args.output_file}")

if __name__ == "__main__":
    main()
