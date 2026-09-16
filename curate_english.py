#!/usr/bin/env python3
import os
import gc
import json
import time
import sys
import logging
import argparse
from typing import List, Optional, Tuple
from enum import Enum
from pydantic import BaseModel, Field

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("english_curation")

# =============================================================================
# Pydantic Schemas
# =============================================================================

class ReasoningLevel(str, Enum):
    L0 = "L0"
    L1 = "L1"
    L2 = "L2"
    L3 = "L3"
    L4 = "L4"
    L5 = "L5"

class ErrorCategory(str, Enum):
    FACTUAL = "factual"
    LOGICAL = "logical"
    ARITHMETIC = "arithmetic"
    COMPREHENSION = "comprehension"
    NONE = "none"
    OTHER = "other"

class Triple(BaseModel):
    subject: str = Field(description="Subject entity")
    relation: str = Field(description="Relation or predicate linking subject and object")
    object: str = Field(description="Object entity")

class EnglishCurationOutput(BaseModel):
    reasoning_level: ReasoningLevel = Field(description="The reasoning level (L0-L5) required to answer the question based on the passage")
    error_category: ErrorCategory = Field(description="Typical/potential error category if the reasoner fails")
    triples: List[Triple] = Field(description="List of subject-relation-object triples decomposed from the passage")

# =============================================================================
# Parsing and Writing Helpers
# =============================================================================

def validate_and_parse(text_response: str) -> Tuple[Optional[dict], Optional[str]]:
    """Validate response against the curation schema."""
    try:
        # Strip code block wrappers if any
        cleaned = text_response.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()
        
        data = json.loads(cleaned)
        validated = EnglishCurationOutput.model_validate(data)
        return validated.model_dump(), None
    except json.JSONDecodeError as jde:
        return None, f"JSON Decode Error: {jde}"
    except Exception as e:
        return None, f"Schema Validation Error: {e}"

def write_success_record(filepath: str, item: dict, parsed_data: dict, triples_str: str, triples_token_length: int, length_matched_passage: str):
    """Write success record immediately to file and flush."""
    merged_record = {
        **item,
        "reasoning_level": parsed_data["reasoning_level"],
        "error_category": parsed_data["error_category"],
        "triples": parsed_data["triples"],
        "triples_str": triples_str,
        "triples_token_length": triples_token_length,
        "length_matched_passage": length_matched_passage
    }
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(merged_record, ensure_ascii=False) + "\n")
        f.flush()
        try:
            os.fsync(f.fileno())
        except Exception:
            pass

def write_failed_record(filepath: str, item: dict, raw_response: str, error_msg: str):
    """Write failed record immediately to file and flush."""
    failed_record = {
        **item,
        "error": error_msg,
        "raw_response": raw_response,
        "failed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(failed_record, ensure_ascii=False) + "\n")
        f.flush()
        try:
            os.fsync(f.fileno())
        except Exception:
            pass

# =============================================================================
# Prompt Construction
# =============================================================================

def build_chat_prompt(tokenizer, item: dict) -> str:
    question_number = item["question_number"]
    passage = item["flores_passage"]
    question = item["question"]
    mc_answers = [
        item["mc_answer1"],
        item["mc_answer2"],
        item["mc_answer3"],
        item["mc_answer4"]
    ]
    correct_answer_num = item["correct_answer_num"]
    
    choices_text = "\n".join([f"{idx+1}: {choice}" for idx, choice in enumerate(mc_answers)])
    
    user_prompt = (
        f"You are an expert linguistic annotator. Analyze the provided passage, question, and choices, and generate the requested metadata JSON.\n\n"
        f"Passage:\n{passage}\n\n"
        f"Question:\n{question}\n\n"
        f"Choices:\n{choices_text}\n\n"
        f"Correct Answer Option: {correct_answer_num}\n\n"
        "INSTRUCTIONS:\n"
        "1. Classify the reasoning level required to answer this question from the passage as one of L0, L1, L2, L3, L4, or L5:\n"
        "   - L0: Direct retrieval / factoid\n"
        "   - L1: Simple inference / single-hop\n"
        "   - L2: Multi-hop reasoning\n"
        "   - L3: Analytical / mathematical reasoning\n"
        "   - L4: Synthesis / evaluative / complex reasoning\n"
        "   - L5: Creative / hypothetical / counterfactual reasoning\n"
        "2. Classify the potential error category that could occur if a model or student fails to reason about this question (factual | logical | arithmetic | comprehension | none | other).\n"
        "3. Decompose the passage into a list of subject-relation-object triples (triples). Each triple must contain:\n"
        "   - subject: The subject entity\n"
        "   - relation: The relationship or verb phrase connecting subject and object\n"
        "   - object: The object entity\n\n"
        "Output MUST be valid JSON strictly adhering to the requested JSON schema."
    )
    
    messages = [
        {"role": "user", "content": user_prompt}
    ]
    
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        try:
            return tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False
            )
        except Exception:
            pass

    return f"<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"

def build_retry_chat_prompt(tokenizer, item: dict, error_msg: str, bad_json: str) -> str:
    question_number = item["question_number"]
    user_prompt = (
        f"You are an expert linguistic annotator. Your previous response for question_number \"{question_number}\" failed JSON schema validation.\n\n"
        f"CRITICAL ERROR ENCOUNTERED:\n{error_msg}\n\n"
        f"PREVIOUS INVALID RESPONSE:\n{bad_json}\n\n"
        "INSTRUCTION: Please re-analyze the inputs, correct the error, and generate a new response "
        "that STRICTLY conforms to the requested JSON schema."
    )

    messages = [
        {"role": "user", "content": user_prompt}
    ]

    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        try:
            return tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False
            )
        except Exception:
            pass

    return f"<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"

# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Curate English dataset using offline vLLM.")
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen2.5-32B-Instruct", help="Path to the vLLM model")
    parser.add_argument("--dataset-file", type=str, default="eng_Latn.jsonl", help="Path to the English dataset")
    parser.add_argument("--output-file", type=str, default="curated_eng_Latn.jsonl", help="Path to output success jsonl file")
    parser.add_argument("--failed-file", type=str, default="failed_eng_Latn.jsonl", help="Path to output failed jsonl file")
    parser.add_argument("--batch-size", type=int, default=64, help="Inference batch size")
    parser.add_argument("--max-model-len", type=int, default=4096, help="Maximum model context length")
    args = parser.parse_args()

    # Ensure parent output directories exist
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    os.makedirs(os.path.dirname(args.failed_file), exist_ok=True)

    if not os.path.exists(args.dataset_file):
        logger.error(f"Dataset file {args.dataset_file} not found.")
        sys.exit(1)

    # 1. Load English Dataset
    logger.info(f"Loading dataset from {args.dataset_file}...")
    all_items = []
    with open(args.dataset_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                all_items.append(json.loads(line))
    
    logger.info(f"Total items in original dataset: {len(all_items)}")
    
    # Sort the 900 questions by len(flores_passage) descending
    all_items.sort(key=lambda x: len(x.get("flores_passage", "")), reverse=True)
    selected_subset = all_items
    logger.info("Loaded full dataset (sorted by passage length).")

    # 2. Resume Logic (using link + question_number for unique matching)
    completed_keys = set()
    for filepath in [args.output_file, args.failed_file]:
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            record = json.loads(line)
                            if "link" in record and "question_number" in record:
                                completed_keys.add((record["link"], int(record["question_number"])))
                        except Exception:
                            pass
    logger.info(f"Resume check complete. Already completed/failed: {len(completed_keys)}")

    # Filter out already processed items from the selected entries
    todo_list = [item for item in selected_subset if (item["link"], int(item["question_number"])) not in completed_keys]

    if not todo_list:
        logger.info("All records processed. Nothing to do!")
        sys.exit(0)

    logger.info(f"Total items remaining to process: {len(todo_list)}")

    # Initialize tokenizer
    from transformers import AutoTokenizer
    logger.info(f"Loading tokenizer from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, local_files_only=True)

    # Define schema for structured output guided decoding
    json_schema = EnglishCurationOutput.model_json_schema()
    
    # Configure guided decoding sampling params
    from vllm import SamplingParams
    guided_decoding_config = None
    max_tokens_val = 2000

    sampling_params_kwargs = {
        "temperature": 0.2,
        "max_tokens": max_tokens_val,
        "repetition_penalty": 1.12
    }

    # try structured_outputs (v0.12.0+)
    try:
        from vllm.sampling_params import StructuredOutputsParams
        guided_decoding_config = StructuredOutputsParams(json=json_schema)
        sampling_params_kwargs["structured_outputs"] = guided_decoding_config
        logger.info("Configured SamplingParams using StructuredOutputsParams.")
    except ImportError:
        pass

    # try guided_decoding from sampling_params
    if guided_decoding_config is None:
        try:
            from vllm.sampling_params import GuidedDecodingParams
            guided_decoding_config = GuidedDecodingParams(json=json_schema)
            sampling_params_kwargs["guided_decoding"] = guided_decoding_config
            logger.info("Configured SamplingParams using GuidedDecodingParams from vllm.sampling_params.")
        except ImportError:
            pass

    # try guided_decoding from protocol
    if guided_decoding_config is None:
        try:
            from vllm.entrypoints.openai.protocol import GuidedDecodingParams
            guided_decoding_config = GuidedDecodingParams(json=json_schema)
            sampling_params_kwargs["guided_decoding"] = guided_decoding_config
            logger.info("Configured SamplingParams using GuidedDecodingParams from vllm.entrypoints.openai.protocol.")
        except ImportError:
            pass

    # try guided_decoding from model_executor
    if guided_decoding_config is None:
        try:
            from vllm.model_executor.guided_decoding import GuidedDecodingParams
            guided_decoding_config = GuidedDecodingParams(json=json_schema)
            sampling_params_kwargs["guided_decoding"] = guided_decoding_config
            logger.info("Configured SamplingParams using GuidedDecodingParams from vllm.model_executor.guided_decoding.")
        except ImportError:
            pass

    # try direct dictionary fallback
    if guided_decoding_config is None:
        try:
            sampling_params_kwargs["guided_decoding"] = {"json": json_schema}
            logger.info("Configured SamplingParams using dictionary guided_decoding fallback.")
        except Exception:
            pass

    sampling_params = SamplingParams(**sampling_params_kwargs)

    # Initialize vLLM Engine
    from vllm import LLM
    logger.info(f"Loading model from {args.model_path} with max_model_len={args.max_model_len}...")
    llm = LLM(
        model=args.model_path,
        tokenizer=args.model_path,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=0.95,
        dtype="bfloat16",
        disable_log_stats=False,
        additional_config={"gdn_prefill_backend": "triton"}
    )

    failed_retry_queue = []
    
    start_time = time.time()
    processed_count = 0
    total_to_process = len(todo_list)

    # Batch loop
    for idx in range(0, total_to_process, args.batch_size):
        batch = todo_list[idx : idx + args.batch_size]
        logger.info("=" * 80)
        logger.info(f"PROCESSING BATCH {idx // args.batch_size + 1}: items {idx + 1} to {idx + len(batch)} of {total_to_process}")
        logger.info("=" * 80)

        vllm_inputs = []
        for item in batch:
            prompt = build_chat_prompt(tokenizer, item)
            vllm_inputs.append({"prompt": prompt})

        # Initial batch generation
        try:
            outputs = llm.generate(vllm_inputs, sampling_params)
        except Exception as e:
            logger.error(f"Error during batch generation: {e}. Writing failures.")
            for item in batch:
                write_failed_record(args.failed_file, item, "", f"Batch generation engine error: {e}")
            
            processed_count += len(batch)
            continue

        # Process results and collect failures for later retry
        for item, out in zip(batch, outputs):
            q_num = item["question_number"]
            text_response = out.outputs[0].text

            # Parse and validate JSON
            parsed_data, error_msg = validate_and_parse(text_response)

            if parsed_data is None:
                logger.warning(f"[q_num: {q_num}] Initial schema validation failed: {error_msg}. Queueing for post-processing retry...")
                item["_initial_raw_response"] = text_response
                item["_initial_error_msg"] = error_msg
                failed_retry_queue.append(item)
            else:
                logger.info(f"[q_num: {q_num}] Successfully validated metadata. Applying Condition C6 length slice...")
                # Length-Matched Control (Condition C6)
                triples_str = "".join([f"{t['subject']} | {t['relation']} | {t['object']}\n" for t in parsed_data["triples"]])
                tokens_triples = tokenizer.encode(triples_str, add_special_tokens=False)
                L = len(tokens_triples)
                
                raw_passage = item["flores_passage"]
                tokens_passage = tokenizer.encode(raw_passage, add_special_tokens=False)
                sliced_tokens = tokens_passage[:L]
                length_matched_passage = tokenizer.decode(sliced_tokens)
                
                write_success_record(
                    args.output_file, 
                    item, 
                    parsed_data, 
                    triples_str, 
                    L, 
                    length_matched_passage
                )
        
        processed_count += len(batch)
        elapsed = time.time() - start_time
        speed = processed_count / elapsed if elapsed > 0 else 0
        eta = (total_to_process - processed_count) / speed if speed > 0 else 0
        logger.info(f"Progress: {processed_count}/{total_to_process} items. Speed: {speed:.2f} items/sec. ETA: {eta:.1f}s")

    # Post-processing 8K retry stage
    if failed_retry_queue:
        logger.info("=" * 80)
        logger.info(f"STARTING RETRY PASS FOR {len(failed_retry_queue)} FAILED ITEMS WITH 8K LIMIT...")
        logger.info("=" * 80)

        # Configure retry SamplingParams with max_tokens=8000
        retry_sampling_params_kwargs = sampling_params_kwargs.copy()
        retry_sampling_params_kwargs["max_tokens"] = 8000
        
        # Adjust structured decoding / guided decoding parameters if present
        if "structured_outputs" in retry_sampling_params_kwargs:
            pass # keeps StructuredOutputsParams configuration
        elif "guided_decoding" in retry_sampling_params_kwargs:
            pass # keeps GuidedDecodingParams configuration
            
        retry_sampling_params = SamplingParams(**retry_sampling_params_kwargs)

        retry_batch_size = 16
        for idx in range(0, len(failed_retry_queue), retry_batch_size):
            retry_batch = failed_retry_queue[idx : idx + retry_batch_size]
            retry_vllm_inputs = []
            for item in retry_batch:
                retry_prompt = build_retry_chat_prompt(
                    tokenizer, 
                    item, 
                    item["_initial_error_msg"], 
                    item["_initial_raw_response"]
                )
                retry_vllm_inputs.append({"prompt": retry_prompt})

            try:
                retry_outputs = llm.generate(retry_vllm_inputs, retry_sampling_params)
            except Exception as e:
                logger.error(f"Error during retry generation: {e}. Writing failures.")
                for item in retry_batch:
                    write_failed_record(args.failed_file, item, "", f"Retry generation engine error: {e}")
                continue

            for item, out in zip(retry_batch, retry_outputs):
                q_num = item["question_number"]
                retry_text_response = out.outputs[0].text
                parsed_data, error_msg = validate_and_parse(retry_text_response)

                if parsed_data is not None:
                    logger.info(f"[q_num: {q_num}] Retry succeeded on 8K limit! Applying Condition C6 length slice...")
                    # Length-Matched Control (Condition C6)
                    triples_str = "".join([f"{t['subject']} | {t['relation']} | {t['object']}\n" for t in parsed_data["triples"]])
                    tokens_triples = tokenizer.encode(triples_str, add_special_tokens=False)
                    L = len(tokens_triples)
                    
                    raw_passage = item["flores_passage"]
                    tokens_passage = tokenizer.encode(raw_passage, add_special_tokens=False)
                    sliced_tokens = tokens_passage[:L]
                    length_matched_passage = tokenizer.decode(sliced_tokens)
                    
                    write_success_record(
                        args.output_file, 
                        item, 
                        parsed_data, 
                        triples_str, 
                        L, 
                        length_matched_passage
                    )
                else:
                    logger.error(f"[q_num: {q_num}] Retry failed on 8K limit. Error: {error_msg}. Writing to failed file.")
                    write_failed_record(
                        args.failed_file, 
                        item, 
                        retry_text_response, 
                        f"Schema validation failed on 8K retry. Error: {error_msg}"
                    )

    logger.info(f"Curation complete! Success file: {args.output_file}, Fail file: {args.failed_file}")

if __name__ == "__main__":
    main()
