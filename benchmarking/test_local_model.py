#!/usr/bin/env python3
import os, sys, json, argparse, logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("model_test")

SAMPLE_PROMPTS = [
    "You are an expert reading comprehension evaluator.\nPASSAGE: Barcelona is a city in Spain.\nQUESTION: In which country is Barcelona located?\n1. France\n2. Spain\n3. Italy\n4. Germany\nRespond ONLY with a raw JSON object: {\"answer\": \"2\", \"reasoning\": \"The passage explicitly states that Barcelona is in Spain.\"}",
    "You are an expert reading comprehension evaluator.\nPASSAGE: eBay acquired Skype for 2.6 billion dollars.\nQUESTION: What company was acquired by eBay?\n1. Google\n2. Skype\n3. Yahoo\n4. Microsoft\nRespond ONLY with a raw JSON object: {\"answer\": \"2\", \"reasoning\": \"The passage states eBay acquired Skype.\"}",
]

def load_hf_model_and_tokenizer(model_path):
    import torch
    import transformers
    from transformers import AutoTokenizer, AutoConfig

    logger.info(f"Loading config from {model_path} ...")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    arch = getattr(config, "architectures", [""])[0]
    model_type = getattr(config, "model_type", "")
    logger.info(f"Detected arch={arch}, model_type={model_type}")

    # Load Tokenizer / Processor
    try:
        from transformers import AutoProcessor
        proc = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        tokenizer = proc.tokenizer if hasattr(proc, "tokenizer") else proc
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    if hasattr(tokenizer, "pad_token_id") and tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load Model Class based on architecture
    model = None
    if hasattr(transformers, "Qwen3_5ForConditionalGeneration") and ("qwen3_5" in model_type or "Qwen3_5" in arch):
        logger.info("Using Qwen3_5ForConditionalGeneration...")
        model = transformers.Qwen3_5ForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
        )
    elif hasattr(transformers, "AutoModelForImageTextToText") and ("qwen3_5" in model_type or "Qwen3_5" in arch):
        try:
            logger.info("Using AutoModelForImageTextToText...")
            model = transformers.AutoModelForImageTextToText.from_pretrained(
                model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
            )
        except Exception:
            pass

    if model is None:
        from transformers import AutoModelForCausalLM
        logger.info("Using AutoModelForCausalLM...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
        )

    return model, tokenizer

def test_hf(model_path):
    import torch
    model, tokenizer = load_hf_model_and_tokenizer(model_path)
    logger.info("HuggingFace Model & Tokenizer Loaded Successfully!")
    
    results = []
    for prompt in SAMPLE_PROMPTS:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_tokens = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id if hasattr(tokenizer, "pad_token_id") else None
            )
        input_len = inputs["input_ids"].shape[1]
        gen = tokenizer.decode(output_tokens[0][input_len:], skip_special_tokens=True)
        results.append(gen)
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--mode", choices=["vllm", "hf", "auto"], default="hf")
    args = parser.parse_args()

    model_name = os.path.basename(args.model_path.rstrip("/"))
    logger.info("=" * 70)
    logger.info(f"SANITY LOAD TEST FOR: {model_name} (mode={args.mode})")
    logger.info("=" * 70)

    try:
        results = test_hf(args.model_path)
        logger.info("Sample Inference Results (HuggingFace):")
        for i, text in enumerate(results, 1):
            logger.info(f"Prompt {i}: {text.strip()}")
        logger.info("=" * 70)
        logger.info(f"SANITY TEST PASSED FOR {model_name} (HuggingFace)!")
        logger.info("=" * 70)
    except Exception as e:
        logger.error(f"HuggingFace load failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
