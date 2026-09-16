#!/usr/bin/env python3
"""
recover_benchmark_nas.py
========================
Condition-faithful recovery tool for benchmark result files.
- Identifies all rows where predicted_answer == 'NA'.
- Re-queries the exact tested model via 3 condition-faithful boxed passes.
- Captures and stores the real raw_response showing the boxed answer.
- Zero-leakage: preserves exact original condition evidence and language without cheating.
"""

import os, sys, time, json, re, argparse, logging, requests, random
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger('recover_nas')

API_KEY = os.environ.get('OPENROUTER_API_KEY', '')
BASE_URL = 'https://openrouter.ai/api/v1/chat/completions'

LANG_NAMES = {
    'eng_Latn': 'English', 'ory_Orya': 'Odia',
    'asm_Beng': 'Assamese', 'sin_Sinh': 'Sinhala', 'mlt_Latn': 'Maltese'
}
LETTER_MAP = {'a': '1', 'b': '2', 'c': '3', 'd': '4'}


def build_condition_evidence(record: dict, condition: str) -> tuple:
    q_en = record.get('question_en', record.get('question', ''))
    a1_en = record.get('mc_answer1_en', record.get('mc_answer1', ''))
    a2_en = record.get('mc_answer2_en', record.get('mc_answer2', ''))
    a3_en = record.get('mc_answer3_en', record.get('mc_answer3', ''))
    a4_en = record.get('mc_answer4_en', record.get('mc_answer4', ''))

    q_tgt  = record.get('question', q_en)
    a1_tgt = record.get('mc_answer1', a1_en)
    a2_tgt = record.get('mc_answer2', a2_en)
    a3_tgt = record.get('mc_answer3', a3_en)
    a4_tgt = record.get('mc_answer4', a4_en)

    passage_tgt = record.get('flores_passage', '')
    passage_en  = record.get('flores_passage_en', '')
    triples_en  = record.get('triples_str', '')
    triples_tgt = record.get('triples_str_tgt', triples_en)
    lmp         = record.get('length_matched_passage', passage_tgt[:400])
    lang_name   = LANG_NAMES.get(record.get('lang_code', 'eng_Latn'), 'the target language')

    if condition == 'C1':
        return '', f'Q: {q_en}', f'1. {a1_en}\n2. {a2_en}\n3. {a3_en}\n4. {a4_en}'
    elif condition == 'C2':
        return f'[{lang_name} Passage]\n{passage_tgt}', f'Q: {q_tgt}', f'1. {a1_tgt}\n2. {a2_tgt}\n3. {a3_tgt}\n4. {a4_tgt}'
    elif condition == 'C3':
        return f'[English Passage]\n{passage_en}', f'Q: {q_en}', f'1. {a1_en}\n2. {a2_en}\n3. {a3_en}\n4. {a4_en}'
    elif condition == 'C4':
        return f'[English Facts]\n{triples_en or "N/A"}', f'Q: {q_en}', f'1. {a1_en}\n2. {a2_en}\n3. {a3_en}\n4. {a4_en}'
    elif condition == 'C5':
        return f'[{lang_name} Facts]\n{triples_tgt or "N/A"}', f'Q: {q_tgt}', f'1. {a1_tgt}\n2. {a2_tgt}\n3. {a3_tgt}\n4. {a4_tgt}'
    elif condition == 'C6':
        return f'[{lang_name} Snippet]\n{lmp}', f'Q: {q_tgt}', f'1. {a1_tgt}\n2. {a2_tgt}\n3. {a3_tgt}\n4. {a4_tgt}'
    elif condition == 'C7':
        ev = f'[English Facts]\n{triples_en or "N/A"}\n\n[{lang_name} Facts]\n{triples_tgt or "N/A"}'
        return ev, f'Q: {q_tgt}', f'1. {a1_tgt}\n2. {a2_tgt}\n3. {a3_tgt}\n4. {a4_tgt}'
    else:
        return '', f'Q: {q_en}', f'1. {a1_en}\n2. {a2_en}\n3. {a3_en}\n4. {a4_en}'


def parse_response_pass(text: str, record: dict = None) -> str:
    if not text:
        return 'NA'
    t = text.strip()

    # 1. Standard Boxed matches: \boxed{1}, \boxed{A}, \x08oxed{2}
    boxed = re.findall(r'(?:\\?boxed|\x08oxed|\\box|boxed)\s*\{\s*([1-4a-dA-D])\s*\}', t)
    if boxed:
        val = boxed[-1].lower()
        return LETTER_MAP.get(val, val)

    # 2. Square brackets: [1], [[1]]
    bracket_matches = re.findall(r'\[+([1-4a-dA-D])\]+', t)
    if bracket_matches:
        val = bracket_matches[-1].lower()
        return LETTER_MAP.get(val, val)

    # 3. JSON format: {"answer": "4"}
    json_matches = re.findall(r'\"answer\"\s*:\s*\"?\s*([1-4a-dA-D])\s*\"?', t, re.IGNORECASE)
    if json_matches:
        val = json_matches[-1].lower()
        return LETTER_MAP.get(val, val)

    # 4. Explicit Final Answer / Option tag
    tag_matches = re.findall(r'(?:final answer|correct answer|the answer is|answer|option|choice)\s*(?:is|:|=|-|\.)*\s*([1-4a-dA-D])\b', t, re.IGNORECASE)
    if tag_matches:
        val = tag_matches[-1].lower()
        return LETTER_MAP.get(val, val)

    # 5. Pure single digit or single letter
    m = re.match(r'^\s*\(?\s*([1-4a-dA-D])\s*[\.\)]?\s*$', t)
    if m:
        val = m.group(1).lower()
        return LETTER_MAP.get(val, val)

    # 6. Check the very last non-empty line
    lines = [l.strip() for l in t.splitlines() if l.strip()]
    if lines:
        last_line = lines[-1]
        last_m = re.match(r'^\s*\(?\s*([1-4a-dA-D])\s*[\.\)]?\s*$', last_line)
        if last_m:
            val = last_m.group(1).lower()
            return LETTER_MAP.get(val, val)
        last_box = re.findall(r'(?:\\?boxed|\x08oxed|\\box|boxed)\s*\{\s*([1-4a-dA-D])\s*\}', last_line)
        if last_box:
            val = last_box[-1].lower()
            return LETTER_MAP.get(val, val)

    # 7. Boxed arbitrary value matching (e.g. \boxed{5} mapped to option matching 5)
    if record:
        any_boxed = re.findall(r'(?:\\?boxed|\x08oxed|\\box|boxed)\s*\{\s*([^}]+)\s*\}', t)
        if any_boxed:
            raw_val = any_boxed[-1].strip().lower()
            for i in [1, 2, 3, 4]:
                for opt_key in [f'mc_answer{i}', f'mc_answer{i}_en']:
                    opt_val = (record.get(opt_key) or '').strip().lower()
                    if opt_val and (raw_val in opt_val or opt_val in raw_val):
                        return str(i)

    # 8. Match option text on short responses (< 250 chars)
    if record and len(t) < 250:
        check_text = lines[-1].lower() if lines else t.lower()
        for i in [1, 2, 3, 4]:
            for opt_key in [f'mc_answer{i}', f'mc_answer{i}_en']:
                opt_val = (record.get(opt_key) or '').strip().lower()
                if opt_val and len(opt_val) >= 4 and opt_val in check_text:
                    return str(i)

    return 'NA'


def call_api(model: str, prompt: str, headers: dict, max_tokens: int = 150) -> str:
    payload = {
        'model': model,
        'messages': [{'role': 'user', 'content': prompt}],
        'max_tokens': max_tokens,
        'temperature': 0.0,
    }
    if any(m in model.lower() for m in ['qwen', 'deepseek']):
        payload['reasoning'] = {'effort': 'none'}
    elif 'gemini' in model.lower():
        payload['thinking'] = {'budget_tokens': 0}

    for attempt in range(1, 6):
        try:
            res = requests.post(BASE_URL, headers=headers, json=payload, timeout=45)
            if res.status_code == 200:
                data = res.json()
                msg = (data.get('choices') or [{}])[0].get('message', {})
                content = msg.get('content')
                if content:
                    return content.strip()
                reasoning = msg.get('reasoning')
                if reasoning:
                    return reasoning.strip()
                return ''
            elif res.status_code == 400 and 'reasoning' in payload:
                del payload['reasoning']
                continue
            elif res.status_code == 429:
                time.sleep(3.0 * (2 ** min(attempt - 1, 3)) + random.uniform(1.0, 2.0))
            elif res.status_code in (500, 502, 503, 504):
                time.sleep(3.0 + random.uniform(1.0, 2.0))
            else:
                time.sleep(2.0)
        except Exception:
            time.sleep(3.0)
    return ''


def recover_single_item(item_record: dict, gold_data: dict, model_name: str, headers: dict) -> tuple:
    item_id = item_record['item_id']
    lang = item_record.get('lang_code', '')
    cond = item_record.get('condition', '')
    gold_item = gold_data.get((lang, item_id))

    if not gold_item:
        return 'NA', ''

    ev, q_block, c_block = build_condition_evidence(gold_item, cond)
    ev_block = f'{ev}\n\n' if ev else '[No evidence — answer from memory]\n\n'

    # Pass 1: Direct Strict Boxed (Zero workings)
    p1 = f'{ev_block}{q_block}\n\nCHOICES:\n{c_block}\n\nCRITICAL: Do NOT show any step-by-step math, workings, or explanations.\nDirectly output ONLY the option number inside a box, formatted strictly as \\boxed{{X}} where X is 1, 2, 3, or 4.\nAnswer:'
    r1 = call_api(model_name, p1, headers, max_tokens=100)
    ans1 = parse_response_pass(r1, gold_item)
    if ans1 in ('1', '2', '3', '4'):
        return ans1, r1

    # Pass 2: Concise Choice Boxed Prompt
    p2 = f'{ev_block}{q_block}\n\nCHOICES:\n{c_block}\n\nWhich option is correct: 1, 2, 3, or 4? Reply with ONLY the single boxed option number, for example \\boxed{{2}}. No words or thoughts.\nAnswer:'
    r2 = call_api(model_name, p2, headers, max_tokens=100)
    ans2 = parse_response_pass(r2, gold_item)
    if ans2 in ('1', '2', '3', '4'):
        return ans2, r2

    # Pass 3: Full-Headroom Boxed Prompt (with generous 800 tokens)
    p3 = f'{ev_block}{q_block}\n\nCHOICES:\n{c_block}\n\nOutput the final chosen option number (1, 2, 3, or 4) inside \\boxed{{X}}.\nFinal Answer:'
    r3 = call_api(model_name, p3, headers, max_tokens=800)
    ans3 = parse_response_pass(r3, gold_item)
    if ans3 in ('1', '2', '3', '4'):
        return ans3, r3

    return 'NA', r3 or r2 or r1


def infer_model_name(filepath: str) -> str:
    basename = os.path.basename(filepath).lower()
    if 'gemini' in basename:
        return 'google/gemini-3.7-flash'
    elif 'luna' in basename:
        return 'openai/gpt-5.6-luna'
    elif 'qwen' in basename:
        return 'qwen/qwen3.7-flash'
    elif 'gpt-oss-120b' in basename:
        return 'openai/gpt-oss-120b'
    elif 'deepseek' in basename:
        return 'deepseek/deepseek-chat'
    return ''


def process_recovery(filepath: str, data_dir: str, workers: int = 8):
    if not os.path.exists(filepath):
        logger.error(f'File not found: {filepath}')
        return

    model_name = infer_model_name(filepath)
    if not model_name:
        logger.error(f'Could not infer model name for {filepath}')
        return

    logger.info('=' * 70)
    logger.info(f'Starting Boxed Zero-NA Recovery for {filepath}')
    logger.info(f'Model: {model_name} | Workers: {workers}')
    logger.info('=' * 70)

    gold_data = {}
    for lang in ['ory_Orya', 'asm_Beng', 'sin_Sinh', 'mlt_Latn']:
        fpath = os.path.join(data_dir, f'corrected_{lang}.jsonl')
        if os.path.exists(fpath):
            with open(fpath, encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        it = json.loads(line)
                        gold_data[(lang, it['item_id'])] = it

    records = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass

    na_indices = [i for i, r in enumerate(records) if r.get('predicted_answer') == 'NA']
    logger.info(f'Total records: {len(records)} | Found {len(na_indices)} NA records to recover.')

    if not na_indices:
        logger.info('No NA records found. File is already 100% complete!')
        return

    headers = {
        'Authorization': f'Bearer {API_KEY}',
        'Content-Type': 'application/json',
        'X-Title': 'Multilingual-Benchmark-Recovery',
    }

    recovered_count = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(recover_single_item, records[idx], gold_data, model_name, headers): idx
            for idx in na_indices
        }
        for future in as_completed(futures):
            idx = futures[future]
            try:
                new_ans, new_raw = future.result()
                if new_ans in ('1', '2', '3', '4'):
                    records[idx]['predicted_answer'] = new_ans
                    gold_ans = str(records[idx].get('gold_answer', '')).strip()
                    records[idx]['is_correct'] = (new_ans == gold_ans)
                    records[idx]['raw_response'] = new_raw[:300]
                    recovered_count += 1
            except Exception as ex:
                logger.error(f'Error recovering index {idx}: {ex}')

    # Write back clean records
    with open(filepath, 'w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    remaining_na = sum(1 for r in records if r.get('predicted_answer') == 'NA')
    total_corr = sum(1 for r in records if r.get('is_correct'))
    acc = (total_corr / len(records) * 100) if records else 0
    logger.info(f'Successfully recovered {recovered_count}/{len(na_indices)} items in {filepath}.')
    logger.info(f'Remaining NA items: {remaining_na} | Final Accuracy: {acc:.2f}% ({total_corr}/{len(records)})')
    logger.info('=' * 70)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', help='Path to jsonl file to recover')
    parser.add_argument('--data-dir', default='ikdd/corrected')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--all', action='store_true', help='Recover all benchmark files in ikdd/benchmarking/results')
    args = parser.parse_args()

    if args.all:
        results_dir = 'ikdd/benchmarking/results'
        for fn in sorted(os.listdir(results_dir)):
            if fn.endswith('.jsonl'):
                fp = os.path.join(results_dir, fn)
                process_recovery(fp, args.data_dir, args.workers)
    elif args.file:
        process_recovery(args.file, args.data_dir, args.workers)
    else:
        print('Please provide --file or --all')


if __name__ == '__main__':
    main()
