#!/usr/bin/env python3
"""
generate_synthetic.py - Dual-Pass Synthetic Question Generation Pipeline
Pass 1: Model generates question with thinking enabled
Pass 2: Model independently verifies/overrides the correct answer with thinking
Quality guards: strict error_category whitelist, answer position shuffling, hard validators
"""
import os, json, time, sys, random, logging, argparse
from typing import List, Optional, Tuple
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("synthetic_generation")

VALID_ERROR_CATEGORIES = {"factual", "logical", "arithmetic", "comprehension", "none"}

# =============================================================================
# Pydantic Schemas
# =============================================================================

class Triple(BaseModel):
    subject: str = Field(description="Subject entity")
    relation: str = Field(description="Relation or predicate")
    object: str = Field(description="Object entity")

class GenerationOutput(BaseModel):
    question: str = Field(description="The generated multiple-choice question")
    mc_answer1: str = Field(description="Choice 1")
    mc_answer2: str = Field(description="Choice 2")
    mc_answer3: str = Field(description="Choice 3")
    mc_answer4: str = Field(description="Choice 4")
    correct_answer_num: str = Field(description="1-indexed correct choice: '1','2','3','4'")
    error_category: str = Field(description="factual | logical | arithmetic | comprehension | none")
    triples: List[Triple] = Field(description="Subject-relation-object triples from the passage")

class VerificationOutput(BaseModel):
    verified_correct_answer_num: str = Field(
        description="VERIFIED correct answer: '1','2','3','4' — strictly supported by passage")
    reasoning: str = Field(
        description="Step-by-step reasoning citing exact passage phrases for each choice")
    confidence: str = Field(description="'high', 'medium', or 'low'")

# =============================================================================
# Hard Validators
# =============================================================================

def validate_generation(parsed: dict) -> Tuple[bool, str]:
    for field in ["question","mc_answer1","mc_answer2","mc_answer3","mc_answer4",
                  "correct_answer_num","error_category"]:
        if not parsed.get(field,"").strip():
            return False, f"Empty required field: '{field}'"
    if parsed["correct_answer_num"].strip() not in {"1","2","3","4"}:
        return False, f"Invalid correct_answer_num: '{parsed['correct_answer_num']}'"
    if parsed["error_category"].strip().lower() not in VALID_ERROR_CATEGORIES:
        return False, f"Invalid error_category: '{parsed['error_category']}'"
    if len(parsed.get("triples",[])) < 3:
        return False, f"Too few triples: {len(parsed.get('triples',[]))}"
    choices = [parsed[f"mc_answer{i}"] for i in range(1,5)]
    if len(set(c.strip().lower() for c in choices)) < 4:
        return False, "Duplicate choices detected"
    return True, ""

def validate_verification(parsed: dict) -> Tuple[bool, str]:
    if parsed.get("verified_correct_answer_num","").strip() not in {"1","2","3","4"}:
        return False, f"Invalid verified_correct_answer_num: '{parsed.get('verified_correct_answer_num','')}'"
    if not parsed.get("reasoning","").strip():
        return False, "Empty reasoning"
    if parsed.get("confidence","").strip() not in {"high","medium","low"}:
        return False, f"Invalid confidence: '{parsed.get('confidence','')}'"
    return True, ""

# =============================================================================
# JSON Parsing
# =============================================================================

def parse_json_response(text: str, model_cls) -> Tuple[Optional[dict], Optional[str]]:
    try:
        cleaned = text.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()
        # Strip think blocks
        if "<think>" in cleaned:
            think_end = cleaned.rfind("</think>")
            if think_end != -1:
                cleaned = cleaned[think_end + len("</think>"):].strip()
            elif cleaned.startswith("<think>"):
                cleaned = cleaned[cleaned.find(">") + 1:].strip()
        # Find first {
        start = cleaned.find("{")
        if start != -1:
            cleaned = cleaned[start:]
        data = json.loads(cleaned)
        validated = model_cls.model_validate(data)
        return validated.model_dump(), None
    except json.JSONDecodeError as jde:
        return None, f"JSON Decode Error: {jde}"
    except Exception as e:
        return None, f"Schema Validation Error: {e}"

# =============================================================================
# File Writers
# =============================================================================

def write_success_record(filepath, task, gen_data, verif_data, triples_str, L, lmp):
    record = {
        "synth_id": task["synth_id"],
        "link": task["link"],
        "flores_passage": task["flores_passage"],
        "question": gen_data["question"],
        "mc_answer1": gen_data["mc_answer1"],
        "mc_answer2": gen_data["mc_answer2"],
        "mc_answer3": gen_data["mc_answer3"],
        "mc_answer4": gen_data["mc_answer4"],
        "correct_answer_num": verif_data["verified_correct_answer_num"],
        "dialect": task["dialect"],
        "ds": task["ds"],
        "reasoning_level": task["target_level"],
        "error_category": gen_data["error_category"].strip().lower(),
        "triples": gen_data["triples"],
        "triples_str": triples_str,
        "triples_token_length": L,
        "length_matched_passage": lmp,
        "generation_proposed_answer": gen_data["correct_answer_num"],
        "verification_reasoning": verif_data["reasoning"],
        "verification_confidence": verif_data["confidence"],
        "answer_was_corrected": gen_data["correct_answer_num"].strip() != verif_data["verified_correct_answer_num"].strip()
    }
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        try: os.fsync(f.fileno())
        except: pass

def write_failed_record(filepath, task, raw_response, error_msg):
    rec = {
        "synth_id": task["synth_id"],
        "link": task["link"],
        "flores_passage": task["flores_passage"],
        "target_level": task["target_level"],
        "dialect": task["dialect"],
        "ds": task["ds"],
        "error": error_msg,
        "raw_response": (raw_response or "")[:2000],
        "failed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()
        try: os.fsync(f.fileno())
        except: pass

# =============================================================================
# Prompt Construction
# =============================================================================

LEVEL_DESCRIPTIONS = {
    "L2": "L2 - Multi-hop Reasoning: Link 2+ distinct facts from DIFFERENT sentences. One sentence alone cannot answer the question.",
    "L3": "L3 - Analytical/Quantitative Reasoning: Requires arithmetic, percentages, date calculations, or logical counting from numbers explicitly in the passage.",
    "L4": "L4 - Evaluative/Inferential Reasoning: The correct answer is NOT stated verbatim; the reader must evaluate a claim or infer an unstated conclusion that logically follows.",
    "L5": "L5 - Counterfactual/Hypothetical Reasoning: Presents a 'What if X had not occurred?' scenario; reader must reason about consequences using passage content.",
}

def build_generation_prompt(tokenizer, passage: str, target_level: str) -> str:
    desc = LEVEL_DESCRIPTIONS[target_level]
    system_msg = (
        "You are an expert psychometrician and reading comprehension test designer. "
        "CRITICAL RULE: Randomize the correct answer position — distribute correct answers "
        "uniformly across positions 1, 2, 3, and 4. Do NOT default to position 2."
    )
    user_msg = (
        f"Generate a multiple-choice question at difficulty level:\n  {desc}\n\n"
        f"PASSAGE:\n{passage}\n\n"
        "REQUIREMENTS:\n"
        f"1. Write one question testing {target_level} reasoning.\n"
        "2. Write 4 plausible choices — distractors must be believable, not trivially wrong.\n"
        "3. correct_answer_num: '1','2','3', or '4'. RANDOMIZE the position each time.\n"
        "4. error_category: exactly one of: factual | logical | arithmetic | comprehension | none\n"
        "5. At least 5 subject-relation-object triples from the passage.\n\n"
        "Think step-by-step first. Then output ONLY valid JSON. No markdown fences."
    )
    messages = [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}]
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

def build_verification_prompt(tokenizer, passage, question, a1, a2, a3, a4, proposed) -> str:
    system_msg = (
        "You are an expert reading comprehension verifier. Verify which answer is STRICTLY "
        "supported by the passage — only by what the passage says, not general knowledge."
    )
    user_msg = (
        f"PASSAGE:\n{passage}\n\n"
        f"QUESTION:\n{question}\n\n"
        f"CHOICES:\n  1. {a1}\n  2. {a2}\n  3. {a3}\n  4. {a4}\n\n"
        f"Proposed answer: {proposed}\n\n"
        "TASK:\n"
        "1. For each choice, cite exact passage phrases that support or rule it out.\n"
        "2. Identify the single correct answer strictly from passage evidence.\n"
        "3. verified_correct_answer_num: '1','2','3', or '4'\n"
        "4. confidence: 'high' (unambiguous), 'medium' (mild ambiguity), 'low' (unclear)\n\n"
        "Think step-by-step. Output ONLY valid JSON. No markdown fences."
    )
    messages = [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}]
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

# =============================================================================
# Answer Position Shuffler
# =============================================================================

def shuffle_answer_position(gen_data: dict, rng: Optional[random.Random] = None) -> dict:
    """Randomly permute choices so correct answer lands in a uniform random position."""
    if rng is None:
        rng = random.Random(42)
    choices = [gen_data[f"mc_answer{i}"] for i in range(1, 5)]
    correct_idx = int(gen_data["correct_answer_num"]) - 1

    assert len(set(choices)) == 4, "Duplicate answer options detected"
    assert 0 <= correct_idx < 4, f"Invalid correct_idx: {correct_idx}"

    order = list(range(4))
    rng.shuffle(order)

    result = dict(gen_data)
    for new_pos, old_pos in enumerate(order, start=1):
        result[f"mc_answer{new_pos}"] = choices[old_pos]

    result["correct_answer_num"] = str(order.index(correct_idx) + 1)
    return result

# =============================================================================
# SamplingParams
# =============================================================================

def make_sampling_params(json_schema, max_tokens: int, temperature: float = 0.6):
    from vllm import SamplingParams
    kwargs = {"temperature": temperature, "max_tokens": max_tokens, "top_p": 0.95, "top_k": 20}
    guided = None
    try:
        from vllm.sampling_params import StructuredOutputsParams
        guided = StructuredOutputsParams(json=json_schema)
        kwargs["structured_outputs"] = guided
        logger.info("Using StructuredOutputsParams.")
    except ImportError: pass
    if guided is None:
        try:
            from vllm.sampling_params import GuidedDecodingParams
            guided = GuidedDecodingParams(json=json_schema)
            kwargs["guided_decoding"] = guided
            logger.info("Using GuidedDecodingParams.")
        except ImportError: pass
    if guided is None:
        try:
            kwargs["guided_decoding"] = {"json": json_schema}
            logger.info("Using dict guided_decoding fallback.")
        except: pass
    return SamplingParams(**kwargs)

# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Dual-pass synthetic L2-L5 data generation.")
    parser.add_argument("--model-path", type=str,
                        default="Qwen/Qwen2.5-32B-Instruct")
    parser.add_argument("--curated-file", type=str, default="curated_eng_Latn.jsonl")
    parser.add_argument("--output-file", type=str, default="synthetic_data_v2.jsonl")
    parser.add_argument("--failed-file", type=str, default="failed_synthetic_data_v2.jsonl")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--verify-batch-size", type=int, default=16)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gen-max-tokens", type=int, default=4096)
    parser.add_argument("--verif-max-tokens", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    for filepath in [args.output_file, args.failed_file]:
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

    if not os.path.exists(args.curated_file):
        logger.error(f"Curated file not found: {args.curated_file}")
        sys.exit(1)

    # 1. Load unique passages
    logger.info(f"Loading curated dataset from {args.curated_file}...")
    curated_items = []
    with open(args.curated_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                curated_items.append(json.loads(line))

    seen, unique_passages = set(), []
    for item in curated_items:
        p = item["flores_passage"]
        if p not in seen:
            seen.add(p)
            unique_passages.append({"flores_passage": p, "link": item.get("link",""),
                                    "dialect": item.get("dialect","eng_Latn"), "ds": item.get("ds","")})
    logger.info(f"Extracted {len(unique_passages)} unique passages from {len(curated_items)} items.")

    # 2. Build tasks
    shuffled_passages = unique_passages[:]
    random.shuffle(shuffled_passages)
    levels = ["L2","L3","L4","L5"]
    tasks = []
    for lvl in levels:
        for count in range(200):
            p = shuffled_passages[count % len(shuffled_passages)]
            tasks.append({"flores_passage": p["flores_passage"], "link": p["link"],
                          "dialect": p["dialect"], "ds": p["ds"],
                          "target_level": lvl, "synth_id": f"{lvl}_{count+1}"})
    logger.info(f"Created {len(tasks)} tasks (200 per level x 4 levels).")

    # 3. Resume
    completed_ids = set()
    for filepath in [args.output_file, args.failed_file]:
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            r = json.loads(line)
                            if "synth_id" in r:
                                completed_ids.add(r["synth_id"])
                        except: pass
    logger.info(f"Resume: {len(completed_ids)} already done.")

    todo_list = [t for t in tasks if t["synth_id"] not in completed_ids]
    if not todo_list:
        logger.info("All tasks completed!")
        sys.exit(0)
    logger.info(f"Tasks remaining: {len(todo_list)}")

    # 4. Init tokenizer + LLM
    from transformers import AutoTokenizer
    from vllm import LLM
    logger.info(f"Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, local_files_only=True)
    logger.info(f"Loading model with max_model_len={args.max_model_len}...")
    llm = LLM(model=args.model_path, tokenizer=args.model_path, trust_remote_code=True,
              max_model_len=args.max_model_len, gpu_memory_utilization=0.95, dtype="bfloat16",
              disable_log_stats=False, additional_config={"gdn_prefill_backend": "triton"})

    gen_params = make_sampling_params(GenerationOutput.model_json_schema(),
                                       max_tokens=args.gen_max_tokens, temperature=0.7)
    verif_params = make_sampling_params(VerificationOutput.model_json_schema(),
                                         max_tokens=args.verif_max_tokens, temperature=0.3)

    # =========================================================================
    # PASS 1 — Generation with Thinking
    # =========================================================================
    logger.info("=" * 80)
    logger.info("STARTING PASS 1 — GENERATION WITH THINKING")
    logger.info("=" * 80)

    pass1_success, pass1_failed_ids = [], []
    total = len(todo_list)
    processed, start_time = 0, time.time()

    for batch_start in range(0, total, args.batch_size):
        batch = todo_list[batch_start: batch_start + args.batch_size]
        logger.info(f"PASS 1 BATCH {batch_start // args.batch_size + 1}: "
                    f"items {batch_start+1}-{batch_start+len(batch)} of {total}")

        vllm_inputs = [{"prompt": build_generation_prompt(tokenizer, t["flores_passage"], t["target_level"])}
                       for t in batch]
        try:
            outputs = llm.generate(vllm_inputs, gen_params)
        except Exception as e:
            logger.error(f"Pass1 batch engine error: {e}")
            for t in batch:
                write_failed_record(args.failed_file, t, "", f"Pass1 engine error: {e}")
                pass1_failed_ids.append(t["synth_id"])
            processed += len(batch)
            continue

        for task, out in zip(batch, outputs):
            raw = out.outputs[0].text
            gen_data, err = parse_json_response(raw, GenerationOutput)
            if gen_data is None:
                logger.warning(f"[{task['synth_id']}] Pass1 parse failed: {err}")
                write_failed_record(args.failed_file, task, raw, f"Pass1 parse: {err}")
                pass1_failed_ids.append(task["synth_id"])
                continue
            ok, reason = validate_generation(gen_data)
            if not ok:
                logger.warning(f"[{task['synth_id']}] Pass1 validation failed: {reason}")
                write_failed_record(args.failed_file, task, raw, f"Pass1 validation: {reason}")
                pass1_failed_ids.append(task["synth_id"])
                continue
            task_rng = random.Random(f"{args.seed}_{task['synth_id']}")
            gen_data = shuffle_answer_position(gen_data, rng=task_rng)
            logger.info(f"[{task['synth_id']}] Pass1 OK => proposed answer: {gen_data['correct_answer_num']}")
            pass1_success.append((task, gen_data))

        processed += len(batch)
        elapsed = time.time() - start_time
        speed = processed / elapsed if elapsed > 0 else 0
        eta = (total - processed) / speed if speed > 0 else 0
        logger.info(f"Pass1 Progress: {processed}/{total} | {speed:.2f}/s | ETA: {eta:.0f}s")

    logger.info(f"PASS 1 complete. Success: {len(pass1_success)} | Failed: {len(pass1_failed_ids)}")

    if not pass1_success:
        logger.error("No Pass 1 successes — aborting.")
        sys.exit(1)

    # =========================================================================
    # PASS 2 — Verification with Thinking
    # =========================================================================
    logger.info("=" * 80)
    logger.info(f"STARTING PASS 2 — VERIFICATION WITH THINKING ({len(pass1_success)} items)")
    logger.info("=" * 80)

    verified_count = corrected_count = verif_failed_count = low_conf_count = 0

    for batch_start in range(0, len(pass1_success), args.verify_batch_size):
        vbatch = pass1_success[batch_start: batch_start + args.verify_batch_size]
        logger.info(f"PASS 2 BATCH {batch_start // args.verify_batch_size + 1}: "
                    f"items {batch_start+1}-{batch_start+len(vbatch)} of {len(pass1_success)}")

        verif_inputs = []
        for (task, gen_data) in vbatch:
            p = build_verification_prompt(
                tokenizer, task["flores_passage"], gen_data["question"],
                gen_data["mc_answer1"], gen_data["mc_answer2"],
                gen_data["mc_answer3"], gen_data["mc_answer4"],
                gen_data["correct_answer_num"]
            )
            verif_inputs.append({"prompt": p})

        try:
            verif_outputs = llm.generate(verif_inputs, verif_params)
        except Exception as e:
            logger.error(f"Pass2 batch engine error: {e}")
            for (task, _) in vbatch:
                write_failed_record(args.failed_file, task, "", f"Pass2 engine error: {e}")
                verif_failed_count += 1
            continue

        for (task, gen_data), vout in zip(vbatch, verif_outputs):
            s_id = task["synth_id"]
            raw_v = vout.outputs[0].text
            verif_data, verif_err = parse_json_response(raw_v, VerificationOutput)

            if verif_data is None:
                logger.warning(f"[{s_id}] Pass2 parse failed: {verif_err} — discarding")
                write_failed_record(args.failed_file, task, raw_v, f"Pass2 parse: {verif_err}")
                verif_failed_count += 1
                continue

            ok, reason = validate_verification(verif_data)
            if not ok:
                logger.warning(f"[{s_id}] Pass2 validation failed: {reason} — discarding")
                write_failed_record(args.failed_file, task, raw_v, f"Pass2 validation: {reason}")
                verif_failed_count += 1
                continue

            prop = gen_data["correct_answer_num"].strip()
            verif = verif_data["verified_correct_answer_num"].strip()
            conf = verif_data["confidence"]

            if prop != verif:
                logger.info(f"[{s_id}] ANSWER CORRECTED: {prop} -> {verif} (conf: {conf})")
                corrected_count += 1
            else:
                logger.info(f"[{s_id}] Answer confirmed: {verif} (conf: {conf})")

            if conf == "low":
                low_conf_count += 1
                logger.warning(f"[{s_id}] Low-confidence verification — flagged in output")

            # C6 length-matched passage
            triples_str = "".join(f"{t['subject']} | {t['relation']} | {t['object']}\n"
                                   for t in gen_data["triples"])
            tokens_triples = tokenizer.encode(triples_str, add_special_tokens=False)
            L = len(tokens_triples)
            tokens_passage = tokenizer.encode(task["flores_passage"], add_special_tokens=False)
            lmp = tokenizer.decode(tokens_passage[:L])

            write_success_record(args.output_file, task, gen_data, verif_data, triples_str, L, lmp)
            verified_count += 1

    # =========================================================================
    # Summary
    # =========================================================================
    logger.info("=" * 80)
    logger.info("DUAL-PASS GENERATION COMPLETE")
    logger.info(f"  Pass1 success:          {len(pass1_success)}")
    logger.info(f"  Pass1 failed:           {len(pass1_failed_ids)}")
    logger.info(f"  Pass2 verified output:  {verified_count}")
    logger.info(f"  Pass2 failed:           {verif_failed_count}")
    logger.info(f"  Answers corrected:      {corrected_count}")
    logger.info(f"  Low-confidence items:   {low_conf_count}")
    logger.info(f"  Output: {args.output_file}")
    logger.info("=" * 80)

    # Quick distribution check
    try:
        from collections import Counter
        dist, lvl_dist = Counter(), {l: Counter() for l in ["L2","L3","L4","L5"]}
        with open(args.output_file) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    dist[r["correct_answer_num"]] += 1
                    lvl_dist.get(r.get("reasoning_level","?"), Counter())[r["correct_answer_num"]] += 1
        logger.info("Final answer distribution:")
        for k in ["1","2","3","4"]:
            logger.info(f"  Answer {k}: {dist.get(k,0)}")
        for lvl in ["L2","L3","L4","L5"]:
            logger.info(f"  {lvl}: {dict(lvl_dist[lvl])}")
    except Exception as e:
        logger.warning(f"Distribution check error: {e}")

if __name__ == "__main__":
    main()
