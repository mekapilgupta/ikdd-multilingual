#!/usr/bin/env python3
"""
translate_judge.py — Standalone LLM Judge for Flagged Translations
===================================================================
Runs in a dedicated clean Python process to avoid CUDA state pollution from
prior HF transformers / PyTorch models. Matches the exact vLLM initialization
from ikdd/generate_synthetic.py and ikdd/curate_english.py.
"""
import os, gc, json, sys, time, logging, argparse
from typing import List, Optional
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("translate_judge")

LANG_CONFIG = {
    "ory_Orya": {"name": "Odia"},
    "asm_Beng": {"name": "Assamese"},
    "sin_Sinh": {"name": "Sinhala"},
    "mlt_Latn": {"name": "Maltese"},
}

def build_judge_prompt(tokenizer, original_en: str, trans_a: str, trans_b: str,
                        back_a: str, back_b: str, chrf_a: float, chrf_b: float,
                        labse_a: float, labse_b: float, tgt_lang_name: str) -> str:
    system_msg = (
        "You are an expert multilingual translation quality evaluator. "
        "Your task is to select the better translation based ONLY on faithfulness "
        "to the original English text — meaning, semantic accuracy and completeness, "
        "not just fluency. /no_think"
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
        "Output ONLY a single line of valid JSON — no extra text:\n"
        '{"winner": "A" or "B" or "REJECT", '
        '"confidence": "high" or "medium" or "low", '
        '"errors_in_loser": "max 10 words describing error in loser"}'
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
    import re
    try:
        raw = text.strip()

        # ── Strategy 0: Strip think tags if present ──────────────────────────
        for open_tag, close_tag in [("<think>", "</think>"), ("<|think|>", "<|/think|>")]:
            if open_tag in raw:
                think_end = raw.rfind(close_tag)
                if think_end != -1:
                    raw = raw[think_end + len(close_tag):].strip()

        # ── Strategy 1: Try every { ... } block from LAST to FIRST ───────────
        # Model outputs "Thinking Process: ..." then JSON at the END
        # so we search backwards for valid JSON
        brace_positions = [i for i, c in enumerate(raw) if c == "{"]
        for start in reversed(brace_positions):
            end = raw.rfind("}", start)
            if end == -1:
                continue
            candidate = raw[start:end+1]
            try:
                data = json.loads(candidate)
                if data.get("winner") in {"A", "B", "REJECT"}:
                    return data
            except Exception:
                pass

        # ── Strategy 2: Regex — find "winner": "A/B/REJECT" anywhere ─────────
        winner_match = re.search(r'"winner"\s*:\s*"([AB]|REJECT)"', raw)
        if winner_match:
            winner = winner_match.group(1)
            conf_match = re.search(r'"confidence"\s*:\s*"(\w+)"', raw)
            confidence = conf_match.group(1) if conf_match else "low"
            err_match = re.search(r'"errors_in_loser"\s*:\s*"([^"]*)"', raw)
            errors = err_match.group(1) if err_match else ""
            return {"winner": winner, "confidence": confidence, "errors_in_loser": errors}

        # ── Strategy 3: Plain text verdict at end of output ───────────────────
        # Handles "**Verdict: Translation A**" or "I select: A" etc.
        verdict = re.search(
            r"(?:verdict|select|winner|better|choose)[^A-Za-z]*([AB])",
            raw[-800:], re.IGNORECASE
        )
        if verdict:
            return {"winner": verdict.group(1).upper(), "confidence": "low", "errors_in_loser": ""}

        # ── Strategy 4: Last standalone A or B in last 200 chars ─────────────
        tail = re.findall(r"([AB])", raw[-200:])
        if tail:
            return {"winner": tail[-1], "confidence": "low", "errors_in_loser": ""}

        # Log for inspection
        with open("ikdd/judge_parse_failures.log", "a", encoding="utf-8") as _pf:
            _pf.write(f"=== PARSE FAILED ===\n{raw[:600]}\n\n")
        logger.warning(f"Parse failed. Raw (first 200 chars): {raw[:200]!r}")
        return None
    except Exception as ex:
        logger.warning(f"Parse exception: {ex}. Raw (first 200 chars): {text[:200]!r}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Standalone LLM Judge for translations.")
    parser.add_argument("--sampled-file", default="sampled_eng_Latn.jsonl")
    parser.add_argument("--cache-file", default="translate_cache.jsonl")
    parser.add_argument("--scores-file", default="translate_scores.json")
    parser.add_argument("--decisions-file", default="translate_decisions.json")
    parser.add_argument("--output-dir", default="translated")
    parser.add_argument("--judge-model-path", default="Qwen/Qwen2.5-32B-Instruct")
    parser.add_argument("--languages", nargs="+", default=["ory_Orya", "asm_Beng", "sin_Sinh", "mlt_Latn"])
    parser.add_argument("--judge-batch-size", type=int, default=16)
    parser.add_argument("--judge-max-tokens", type=int, default=2048)
    parser.add_argument("--max-model-len", type=int, default=8192)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load sampled items
    items = []
    with open(args.sampled_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line: items.append(json.loads(line))
    item_map = {it["item_id"]: it for it in items}
    logger.info(f"Loaded {len(items)} items from {args.sampled_file}")

    # 2. Load translation cache
    cache = {}
    with open(args.cache_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    e = json.loads(line)
                    cache[(e["item_id"], e["lang_code"], e["pass"])] = e
                except: pass
    logger.info(f"Loaded {len(cache)} cached entries from {args.cache_file}")

    # 3. Load scores and auto decisions
    if not os.path.exists(args.scores_file) or not os.path.exists(args.decisions_file):
        logger.error(f"Scores or decisions file missing. Run translate_pipeline.py first.")
        sys.exit(1)

    with open(args.scores_file, encoding="utf-8") as f:
        scores_raw = json.load(f)
    scores = {(k.split("|")[0], k.split("|")[1]): v for k, v in scores_raw.items()}

    with open(args.decisions_file, encoding="utf-8") as f:
        decisions_raw = json.load(f)
    decisions = {(k.split("|")[0], k.split("|")[1]): v for k, v in decisions_raw.items()}

    flagged = [(item_map[item_id], lang_code) for (item_id, lang_code), d in decisions.items() if not d.get("auto", True)]
    logger.info(f"Total pairs: {len(decisions)} | Auto-accepted: {len(decisions) - len(flagged)} | Flagged for Judge: {len(flagged)}")

    # 4. If flagged items exist, run vLLM Judge in this dedicated process
    if flagged:
        logger.info("=" * 70)
        logger.info(f"RUNNING vLLM JUDGE on {len(flagged)} flagged items")
        logger.info("=" * 70)
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        logger.info(f"Loading tokenizer from {args.judge_model_path}...")
        judge_tok = AutoTokenizer.from_pretrained(args.judge_model_path, trust_remote_code=True, local_files_only=True)
        
        import logging as _logging
        os.environ["VLLM_LOGGING_LEVEL"] = "DEBUG"
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        # Route all vLLM logs to a dedicated file for deep inspection
        _vllm_log_path = "ikdd/vllm_judge.log"
        _vllm_fh = _logging.FileHandler(_vllm_log_path, mode="w", encoding="utf-8")
        _vllm_fh.setLevel(_logging.DEBUG)
        _vllm_fh.setFormatter(_logging.Formatter("%(asctime)s [%(name)s %(levelname)s] %(message)s"))
        for _n in ["vllm", "vllm.core", "vllm.engine", "vllm.worker", "vllm.executor",
                   "vllm.v1", "vllm.v1.engine", "vllm.v1.engine.core_client"]:
            _log = _logging.getLogger(_n)
            _log.setLevel(_logging.DEBUG)
            _log.addHandler(_vllm_fh)
        # Also add to root so child processes log to it
        _root = _logging.getLogger()
        _root.addHandler(_vllm_fh)
        logger.info(f"Full vLLM logs will be written to: {_vllm_log_path}")
        logger.info(f"Initializing vLLM Engine (matching generate_synthetic.py)...")
        judge_llm = LLM(
            model=args.judge_model_path,
            tokenizer=args.judge_model_path,
            trust_remote_code=True,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=0.95,
            dtype="bfloat16",
            disable_log_stats=False,
            additional_config={"gdn_prefill_backend": "triton"}
        )
        judge_params = SamplingParams(temperature=0.3, max_tokens=args.judge_max_tokens, top_p=0.9, top_k=20)

        for batch_start in range(0, len(flagged), args.judge_batch_size):
            batch = flagged[batch_start: batch_start + args.judge_batch_size]
            prompts = []
            for (item, lang_code) in batch:
                sc = scores.get((item["item_id"], lang_code), {})
                trans_a = cache.get((item["item_id"], lang_code, "1A"), {}).get("translations", {}).get("flores_passage", "")
                trans_b = cache.get((item["item_id"], lang_code, "1B"), {}).get("translations", {}).get("flores_passage", "")
                back_a = cache.get((item["item_id"], lang_code, "back_1A"), {}).get("back_passage", "")
                back_b = cache.get((item["item_id"], lang_code, "back_1B"), {}).get("back_passage", "")
                prompts.append({"prompt": build_judge_prompt(
                    judge_tok, item["flores_passage"], trans_a, trans_b,
                    back_a, back_b, sc.get("chrf_a", 0), sc.get("chrf_b", 0),
                    sc.get("labse_a", 0), sc.get("labse_b", 0),
                    LANG_CONFIG.get(lang_code, {}).get("name", lang_code)
                )})

            try:
                outputs = judge_llm.generate(prompts, judge_params)
            except Exception as e:
                logger.error(f"Judge batch error: {e}")
                for (item, lang_code) in batch:
                    decisions[(item["item_id"], lang_code)] = {"winner": "A", "auto": False, "judge_error": str(e)}
                continue
            
            # Retry failed parses with larger token budget
            retry_indices = []
            retry_prompts = []
            for idx, ((item, lang_code), out) in enumerate(zip(batch, outputs)):
                parsed = parse_judge_response(out.outputs[0].text)
                if parsed is None:
                    retry_indices.append(idx)
                    retry_prompts.append(prompts[idx])
            
            if retry_indices:
                retry_params = SamplingParams(temperature=0.0, max_tokens=1024, top_p=1.0)
                logger.info(f"Retrying {len(retry_indices)} failed parses with max_tokens=1024...")
                try:
                    retry_outputs = judge_llm.generate(retry_prompts, retry_params)
                    for ri, rout in zip(retry_indices, retry_outputs):
                        outputs[ri] = rout
                except Exception as e:
                    logger.warning(f"Retry batch error: {e}")

            for (item, lang_code), out in zip(batch, outputs):
                raw = out.outputs[0].text
                parsed = parse_judge_response(raw)
                key = (item["item_id"], lang_code)
                if parsed:
                    winner = parsed.get("winner", "A")
                    if winner == "REJECT": winner = "A"
                    decisions[key] = {
                        "winner": winner, "auto": False,
                        "judge_confidence": parsed.get("confidence", "low"),
                        "judge_errors": parsed.get("errors_in_loser", ""),
                        "judge_accepted": parsed.get("accepted_translation", "")
                    }
                    logger.info(f"[{item['item_id']}][{lang_code}] Judge: {winner} (conf: {parsed.get('confidence','')})")
                else:
                    decisions[key] = {"winner": "A", "auto": False, "judge_parse_failed": True}
                    logger.warning(f"[{item['item_id']}][{lang_code}] Judge parse failed — defaulting to A")

        del judge_llm, judge_tok
        gc.collect()

    # 5. Write Final Translated Files
    logger.info("=" * 70)
    logger.info("WRITING FINAL TRANSLATED DATASET FILES")
    logger.info("=" * 70)
    stats = defaultdict(lambda: {"accepted_A": 0, "accepted_B": 0, "judge_adjudicated": 0})

    for lang_code in args.languages:
        out_path = os.path.join(args.output_dir, f"translated_{lang_code}.jsonl")
        written = 0
        with open(out_path, "w", encoding="utf-8") as fout:
            for item in items:
                key = (item["item_id"], lang_code)
                if key not in decisions: continue
                dec = decisions[key]
                winner = dec.get("winner", "A")
                pass_tag = "1A" if winner == "A" else "1B"
                trans_entry = cache.get((item["item_id"], lang_code, pass_tag), {})
                trans = trans_entry.get("translations", {})
                sc = scores.get(key, {})

                judge_accepted = dec.get("judge_accepted", "")
                final_passage = judge_accepted if judge_accepted else trans.get("flores_passage", "")

                record = {
                    "item_id": item["item_id"],
                    "lang_code": lang_code,
                    "flores_passage_en": item.get("flores_passage", ""),
                    "question_en": item.get("question", ""),
                    "mc_answer1_en": item.get("mc_answer1", ""),
                    "mc_answer2_en": item.get("mc_answer2", ""),
                    "mc_answer3_en": item.get("mc_answer3", ""),
                    "mc_answer4_en": item.get("mc_answer4", ""),
                    "flores_passage": final_passage,
                    "question": trans.get("question", ""),
                    "mc_answer1": trans.get("mc_answer1", ""),
                    "mc_answer2": trans.get("mc_answer2", ""),
                    "mc_answer3": trans.get("mc_answer3", ""),
                    "mc_answer4": trans.get("mc_answer4", ""),
                    "correct_answer_num": item.get("correct_answer_num", ""),
                    "reasoning_level": item.get("reasoning_level", ""),
                    "error_category": item.get("error_category", ""),
                    "link": item.get("link", ""),
                    "ds": item.get("ds", ""),
                    "triples_str": item.get("triples_str", ""),
                    "length_matched_passage": item.get("length_matched_passage", ""),
                    "translation_model_winner": f"1A ({trans_entry.get('model','nllb-3.3B')})" if winner == "A" else "1B (nllb-3.3B)",
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
                if winner == "A": stats[lang_code]["accepted_A"] += 1
                else: stats[lang_code]["accepted_B"] += 1
                if not dec.get("auto", True): stats[lang_code]["judge_adjudicated"] += 1

        logger.info(f"[{lang_code}] Wrote {written} records to {out_path}")
        st = stats[lang_code]
        logger.info(f"  Model A wins: {st['accepted_A']} | Model B wins: {st['accepted_B']} | Judge-adjudicated: {st['judge_adjudicated']}")

    logger.info("=" * 70)
    logger.info(f"TRANSLATION PIPELINE 100% COMPLETE! Output directory: {args.output_dir}")
    logger.info("=" * 70)

if __name__ == "__main__":
    main()
