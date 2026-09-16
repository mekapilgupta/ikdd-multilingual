#!/usr/bin/env python3
"""
benchmark_runner.py — Multilingual MCQ Benchmark Runner
=========================================================
Fast batched evaluation with:
  - Thinking mode disabled (Qwen3 enable_thinking=False)
  - Batch generation (default batch_size=100)
  - Exact dataset schema mapping
  - Resume support

Speed optimizations:
  - vLLM backend (--mode vllm): uses bundled vllm-flash-attn, continuous batching,
    PagedAttention — typically 3-5x faster than HF for batch inference on H100
  - HF backend (--mode hf): Flash Attention 2 if flash_attn is installed
  - max_new_tokens = 80 (JSON answer ~30-50 tokens; was 256)
  - Tasks grouped by condition → similar prompt lengths → less padding waste
  - use_cache=True explicitly
  - Flush every N items instead of every item
  - batch_size default = 64
"""

import os, sys, json, re, argparse, logging
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("benchmark")

# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------
def build_prompt(item, condition):
    question = item.get("question", "")
    options = [
        item.get("mc_answer1", ""),
        item.get("mc_answer2", ""),
        item.get("mc_answer3", ""),
        item.get("mc_answer4", "")
    ]
    options_str = "\n".join([f"{i+1}. {opt}" for i, opt in enumerate(options)])

    evidence = ""
    if condition == "C1":
        evidence = ""
    elif condition == "C2":
        p = item.get("flores_passage") or item.get("passage_native", "")
        evidence = f"PASSAGE:\n{p}\n\n"
    elif condition == "C3":
        p_en = item.get("flores_passage_en") or item.get("passage_en", "")
        evidence = f"PASSAGE (English Translation):\n{p_en}\n\n"
    elif condition == "C4":
        t_en = item.get("triples_str") or item.get("triples_en", "")
        evidence = f"FACT TRIPLES (English):\n{t_en}\n\n"
    elif condition == "C5":
        t_nat = item.get("triples_native") or item.get("triples_str", "")
        evidence = f"FACT TRIPLES (Native Language):\n{t_nat}\n\n"
    elif condition == "C6":
        snip = item.get("length_matched_passage") or item.get("snippet_native") or item.get("flores_passage", "")
        evidence = f"PASSAGE SNIPPET:\n{snip}\n\n"

    system_instruction = (
        "You are an expert reading comprehension evaluator. "
        "Select the single best option (1, 2, 3, or 4). "
        "Give a 10-20 word reason.\n"
        'Respond ONLY with valid JSON: {"answer": "<1|2|3|4>", "reasoning": "<10-20 words>"}'
    )

    return (
        f"{system_instruction}\n\n"
        f"{evidence}"
        f"QUESTION:\n{question}\n\n"
        f"OPTIONS:\n{options_str}\n\n"
        f"JSON RESPONSE:"
    )

# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------
def parse_response(raw_text, item):
    predicted = None
    reasoning = ""

    if not raw_text or not isinstance(raw_text, str):
        return None, "", False

    # Strip <think>...</think> block
    cleaned = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()

    # 1. JSON substring search
    json_match = re.search(
        r'\{\s*"answer"\s*:\s*"?(1|2|3|4|mc_answer[1-4])"?\s*(?:,\s*"reasoning"\s*:\s*"([^"]+)")?\s*\}',
        cleaned
    )
    if json_match:
        val = json_match.group(1)
        predicted = val if val in ["1","2","3","4"] else (re.findall(r"\d+", val) or [None])[0]
        if json_match.group(2):
            reasoning = json_match.group(2).strip()

    # 2. Direct JSON parse
    if predicted is None:
        try:
            data = json.loads(cleaned)
            if isinstance(data, dict):
                raw_ans = str(data.get("answer","")).strip()
                reasoning = str(data.get("reasoning","")).strip()
                if raw_ans in ["1","2","3","4"]:
                    predicted = raw_ans
                elif raw_ans.lower().startswith("mc_answer"):
                    nums = re.findall(r"\d+", raw_ans)
                    if nums and nums[0] in ["1","2","3","4"]:
                        predicted = nums[0]
        except Exception:
            pass

    # 3. Digit fallback
    if predicted is None:
        m = re.search(r'(?:answer|option|choice)?\s*[:=\-]?\s*([1-4])\b', cleaned, re.IGNORECASE)
        if m:
            predicted = m.group(1)

    # 4. Option text match
    if predicted is None:
        for idx in range(1, 5):
            opt_text = item.get(f"mc_answer{idx}", "").strip()
            if opt_text and len(opt_text) > 3 and opt_text.lower() in cleaned.lower():
                predicted = str(idx)
                break

    if not reasoning:
        m_r = re.search(r'(?:reason|reasoning|because)\s*[:=\-]\s*([^\n\.\}]+)', cleaned, re.IGNORECASE)
        if m_r:
            reasoning = m_r.group(1).strip()
        else:
            reasoning = cleaned.split("\n")[0][:200]

    gold = str(item.get("correct_answer_num") or item.get("correct_answer") or "").strip()
    is_correct = (predicted == gold) if predicted is not None and gold else False
    return predicted, reasoning, is_correct

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model_hf(model_path):
    """Load model with HuggingFace Transformers."""
    import transformers
    from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    arch = getattr(config, "architectures", [""])[0]
    model_type = getattr(config, "model_type", "")
    logger.info(f"Model architecture: {arch}, model_type: {model_type}")

    try:
        from transformers import AutoProcessor
        proc = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        tokenizer = proc.tokenizer if hasattr(proc, "tokenizer") else proc
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Left-pad for batched generation
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Try Flash Attention 2
    fa2_kwargs = {}
    try:
        import flash_attn  # noqa
        fa2_kwargs["attn_implementation"] = "flash_attention_2"
        logger.info("Flash Attention 2 enabled.")
    except ImportError:
        logger.info("flash_attn not found — using default attention.")

    model = None
    if hasattr(transformers, "Qwen3_5ForConditionalGeneration") and ("qwen3_5" in model_type or "Qwen3_5" in arch):
        logger.info("Loading Qwen3_5ForConditionalGeneration...")
        model = transformers.Qwen3_5ForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True, **fa2_kwargs
        )
    if model is None:
        logger.info("Loading AutoModelForCausalLM...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True, **fa2_kwargs
        )

    model.eval()
    has_chat = hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None
    supports_thinking = "qwen3" in model_type.lower() or "qwen3" in arch.lower()
    return model, tokenizer, has_chat, supports_thinking


def load_model_vllm(model_path):
    """Load model with vLLM (uses bundled vllm-flash-attn, PagedAttention, continuous batching)."""
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer, AutoConfig

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    arch = getattr(config, "architectures", [""])[0]
    model_type = getattr(config, "model_type", "")
    logger.info(f"Model architecture: {arch}, model_type: {model_type}")

    try:
        from transformers import AutoProcessor
        proc = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        tokenizer = proc.tokenizer if hasattr(proc, "tokenizer") else proc
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    logger.info("Loading model with vLLM (PagedAttention + vllm-flash-attn)...")
    # Note: do NOT pass trust_remote_code=True to LLM() — vLLM 0.19+ has native
    # Qwen3ForCausalLM support and trust_remote_code breaks the model registry subprocess.
    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        gpu_memory_utilization=0.90,
        max_model_len=2200,      # prompt up to 2048 + 80 new tokens + buffer
        enable_prefix_caching=True,   # cache shared prompt prefixes (system instruction)
        enforce_eager=False,          # allow CUDA graph capture for speed
    )

    has_chat = hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None
    supports_thinking = "qwen3" in model_type.lower() or "qwen3" in arch.lower()
    return llm, tokenizer, has_chat, supports_thinking

# ---------------------------------------------------------------------------
# Template application
# ---------------------------------------------------------------------------
def apply_template(tokenizer, prompt, has_chat, supports_thinking):
    if has_chat:
        messages = [{"role": "user", "content": prompt}]
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        if supports_thinking:
            kwargs["enable_thinking"] = False
        return tokenizer.apply_chat_template(messages, **kwargs)
    return prompt

# ---------------------------------------------------------------------------
# Batch inference
# ---------------------------------------------------------------------------
def run_batch_hf(model, tokenizer, prompts, max_new_tokens):
    """HF Transformers batched generation."""
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    )
    enc = {k: v.to(model.device) for k, v in enc.items()}

    with torch.no_grad():
        output_tokens = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    padded_input_len = enc["input_ids"].shape[1]
    responses = []
    for i in range(len(prompts)):
        new_tokens = output_tokens[i][padded_input_len:]
        raw_response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        responses.append(raw_response)
    return responses


def run_batch_vllm(llm, prompts, max_new_tokens):
    """vLLM batched generation — sends all prompts at once, returns in order."""
    from vllm import SamplingParams
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=max_new_tokens,
        skip_special_tokens=True,
    )
    outputs = llm.generate(prompts, sampling_params)
    return [out.outputs[0].text.strip() for out in outputs]

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--lang-code", required=True)
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for HF mode. vLLM ignores this (handles internally).")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=80,
                        help="Max new tokens to generate (JSON answer ~30-50 tokens)")
    parser.add_argument("--mode", default="hf", choices=["hf", "vllm"],
                        help="Inference backend: hf (HuggingFace) or vllm (much faster on H100)")
    parser.add_argument("--flush-every", type=int, default=50,
                        help="Flush output file every N completed items")
    args = parser.parse_args()

    items = []
    with open(args.input_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    logger.info(f"Loaded {len(items)} items from {args.input_file}")

    # Load model
    if args.mode == "vllm":
        llm, tokenizer, has_chat, supports_thinking = load_model_vllm(args.model_path)
        model = None
    else:
        model, tokenizer, has_chat, supports_thinking = load_model_hf(args.model_path)
        llm = None

    # Resume support
    existing_keys = set()
    if os.path.exists(args.output_file):
        with open(args.output_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        if record.get("gold_answer") and record.get("predicted_answer"):
                            existing_keys.add((record.get("item_id"), record.get("condition")))
                    except Exception:
                        pass
        logger.info(f"Resuming: {len(existing_keys)} already done.")

    conditions = ["C1", "C2", "C3", "C4", "C5", "C6"]
    total_evals = len(items) * len(conditions)
    logger.info(f"Total evaluations: {total_evals}")

    # Group tasks by condition to minimize padding waste within each batch
    # (same condition → similar prompt lengths → far less wasted padding compute)
    tasks_by_condition = {cond: [] for cond in conditions}
    for item in items:
        gold_ans = str(item.get("correct_answer_num") or item.get("correct_answer") or "").strip()
        for cond in conditions:
            if (item.get("item_id"), cond) not in existing_keys:
                tasks_by_condition[cond].append((item, cond, gold_ans))

    # Flatten: all C1 first (short prompts), then C2/C3 (long passage prompts), etc.
    tasks = []
    for cond in conditions:
        tasks.extend(tasks_by_condition[cond])

    logger.info(f"Tasks remaining: {len(tasks)}")

    out_handle = open(args.output_file, "a", encoding="utf-8")
    completed_count = len(existing_keys)
    correct_count = 0
    flush_counter = 0

    if args.mode == "vllm":
        # vLLM: send ALL prompts at once — it handles batching/scheduling internally
        logger.info("vLLM mode: building all prompts and running in one shot...")
        all_prompts = [
            apply_template(tokenizer, build_prompt(item, cond), has_chat, supports_thinking)
            for item, cond, _ in tasks
        ]
        logger.info(f"Running vLLM inference on {len(all_prompts)} prompts...")
        all_responses = run_batch_vllm(llm, all_prompts, args.max_tokens)
        logger.info("vLLM inference complete. Writing results...")

        for i, (item, cond, gold_ans) in enumerate(tasks):
            raw_response = all_responses[i]
            pred, reason, is_corr = parse_response(raw_response, item)

            res_obj = {
                "item_id": item.get("item_id", ""),
                "lang_code": args.lang_code,
                "condition": cond,
                "reasoning_level": item.get("reasoning_level", ""),
                "error_category": item.get("error_category", ""),
                "gold_answer": gold_ans,
                "predicted_answer": pred,
                "is_correct": is_corr,
                "reasoning": reason,
                "raw_response": raw_response,
            }
            out_handle.write(json.dumps(res_obj, ensure_ascii=False) + "\n")

            completed_count += 1
            flush_counter += 1
            if is_corr:
                correct_count += 1

            if flush_counter >= args.flush_every:
                out_handle.flush()
                flush_counter = 0

            if completed_count % 500 == 0:
                pct = (correct_count / max(1, completed_count)) * 100
                logger.info(f"Progress: [{completed_count}/{total_evals}] | Acc: {correct_count}/{completed_count} ({pct:.1f}%)")

    else:
        # HF mode: mini-batches
        for batch_start in range(0, len(tasks), args.batch_size):
            batch = tasks[batch_start: batch_start + args.batch_size]

            prompts = [
                apply_template(tokenizer, build_prompt(item, cond), has_chat, supports_thinking)
                for item, cond, _ in batch
            ]
            responses = run_batch_hf(model, tokenizer, prompts, args.max_tokens)

            for i, (item, cond, gold_ans) in enumerate(batch):
                raw_response = responses[i]
                pred, reason, is_corr = parse_response(raw_response, item)

                res_obj = {
                    "item_id": item.get("item_id", ""),
                    "lang_code": args.lang_code,
                    "condition": cond,
                    "reasoning_level": item.get("reasoning_level", ""),
                    "error_category": item.get("error_category", ""),
                    "gold_answer": gold_ans,
                    "predicted_answer": pred,
                    "is_correct": is_corr,
                    "reasoning": reason,
                    "raw_response": raw_response,
                }
                out_handle.write(json.dumps(res_obj, ensure_ascii=False) + "\n")

                completed_count += 1
                flush_counter += 1
                if is_corr:
                    correct_count += 1

                if flush_counter >= args.flush_every:
                    out_handle.flush()
                    flush_counter = 0

            pct = (correct_count / max(1, completed_count)) * 100
            logger.info(f"Progress: [{completed_count}/{total_evals}] | Acc: {correct_count}/{completed_count} ({pct:.1f}%)")

    out_handle.flush()
    out_handle.close()
    logger.info(f"Done! Results saved to: {args.output_file}")

if __name__ == "__main__":
    main()
