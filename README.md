# Multilingual Low-Resource QA & Evidence Grounding Benchmark

Official implementation, dataset release, audit scripts, and evaluation suite for evaluating evidence presentation formats, language representation, and reasoning categories across low-resource multilingual question answering.

---

## 🌟 Overview

This benchmark is designed to systematically disentangle the effects of **evidence presentation format**, **language representation**, and **reasoning task categories** in low-resource multilingual question answering.

### Key Highlights
- **900 Core Items & 4,500 Localized Instances**: Crossing 5 languages (English anchor + 4 low-resource target languages: **Odia**, **Assamese**, **Sinhala**, and **Maltese**).
- **Hybrid Construction**: Combines 319 human-curated anchor questions with 581 synthetic reasoning questions generated under strict schema constraints.
- **7 Grounding Conditions ($C_1$–$C_7$)**: Contrasts closed-book parametric recall ($C_1$), native ($C_2$) and English ($C_3$) continuous prose, native ($C_5$) and English ($C_4$) structured knowledge triples $(s, p, o)$, 50% positional truncation ($C_6$), and cross-lingual dual triples ($C_7$).
- **6 Reasoning Categories ($L_0$–$L_5$)**: Stratified across Direct Surface Retrieval ($L_0$), Single-Hop Deduction ($L_1$), Multi-Hop Synthesis ($L_2$), Quantitative Reasoning ($L_3$), Formal Negation ($L_4$), and Implicit Pragmatics ($L_5$).
- **Rigorous Auditing**: Blind solver screening (92.4% agreement), integer-sampled stratified bilingual human verification (464 instances across 116 core items, $\kappa = 0.82$, 95.5% quality index), and automated semantic audits.
- **75,600 Experimental Trials**: Full factorially matched evaluation across three frontier model snapshots: **Google Gemini 3.7 Flash**, **OpenAI GPT-5.6 Luna**, and **Alibaba Qwen 3.7 Flash**.

---

## 📊 Benchmark Design Matrix

### 1. The Seven Evidence-Grounding Conditions ($C_1$–$C_7$)

| Condition | Evidence Format | Description | Aggregate Acc (95% CI) |
|:---|:---|:---|:---:|
| **$C_1$** | **Closed-Book Parametric** | No external evidence provided; measures baseline parametric recall | $82.23\% \pm 0.72\%$ |
| **$C_2$** | **Native Continuous Passage** | Full continuous prose passage in the target language | $94.69\% \pm 0.42\%$ |
| **$C_3$** | **English Continuous Passage** | Full continuous prose passage in English (cross-lingual pivot) | **$95.94\% \pm 0.37\%$** |
| **$C_4$** | **English Structured Facts** | Document triples $(s, p, o)$ in English | $93.48\% \pm 0.47\%$ |
| **$C_5$** | **Native Structured Facts** | Document triples $(s, p, o)$ in the target language | $92.64\% \pm 0.49\%$ |
| **$C_6$** | **Truncated Native Snippet** | First 50% positional prefix of the target-language passage | $89.07\% \pm 0.59\%$ |
| **$C_7$** | **Cross-Lingual Dual Facts** | Combined English and target-language structured triples | $92.09\% \pm 0.51\%$ |

### 2. The Six Reasoning-Demand Categories ($L_0$–$L_5$)

| Level | Reasoning Task Category | Human Curated | Synthetic | Total Items | Human Audit Sample | Aggregate Acc (95% CI) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **$L_0$** | Direct Surface Retrieval | 150 | 0 | 150 | 0 (0 inst) | $95.00\% \; [93.42, 96.58]$ |
| **$L_1$** | Single-Hop Deduction | 0 | 150 | 150 | 30 (120 inst) | $90.71\% \; [88.05, 93.38]$ |
| **$L_2$** | Multi-Hop Synthesis | 50 | 100 | 150 | 20 (80 inst) | $91.98\% \; [89.15, 94.82]$ |
| **$L_3$** | Quantitative Reasoning | 0 | 150 | 150 | 30 (120 inst) | $84.98\% \; [81.39, 88.56]$ |
| **$L_4$** | Formal Negation | 119 | 31 | 150 | 6 (24 inst) | $94.79\% \; [93.12, 96.46]$ |
| **$L_5$** | Implicit Pragmatics | 0 | 150 | 150 | 30 (120 inst) | $91.23\% \; [88.57, 93.89]$ |
| **Total** | **Benchmark Pool** | **319** | **581** | **900** | **116 (464 inst)** | **$91.45\% \; [90.37, 92.53]$** |

### 3. Language Coverage

| Language | ISO Code | Script | Family | Primary MT Engine | Localized Instances |
|:---|:---:|:---|:---|:---|:---:|
| **English** | `eng_Latn` | Latin | Germanic | Source Anchor | 900 core items |
| **Odia** | `ory_Orya` | Odia | Indo-Aryan | IndicTrans2-1B | 900 instances |
| **Assamese** | `asm_Beng` | Bengali-Assamese | Indo-Aryan | IndicTrans2-1B | 900 instances |
| **Sinhala** | `sin_Sinh` | Sinhala | Insular Indo-Aryan | NLLB-200-3.3B | 900 instances |
| **Maltese** | `mlt_Latn` | Latin | Semitic | NLLB-200-3.3B | 900 instances |

---

## 📁 Repository Structure

```
├── curated_eng_Latn.jsonl            # 319 human-curated English anchor items
├── synthetic_data_v2.jsonl           # 581 dual-pass verified synthetic reasoning items (L1-L5)
├── sampled_eng_Latn.jsonl            # Complete 900-item English benchmark dataset (150 per L0-L5)
│
├── translated/                       # Raw NMT localized datasets (Pre-correction)
│   ├── translated_ory_Orya.jsonl     # Odia translations
│   ├── translated_asm_Beng.jsonl     # Assamese translations
│   ├── translated_sin_Sinh.jsonl     # Sinhala translations
│   └── translated_mlt_Latn.jsonl     # Maltese translations
│
├── corrected/                        # Final post-correction benchmark datasets (900 items each)
│   ├── corrected_ory_Orya.jsonl      # Final Odia benchmark instances
│   ├── corrected_asm_Beng.jsonl      # Final Assamese benchmark instances
│   ├── corrected_sin_Sinh.jsonl      # Final Sinhala benchmark instances
│   ├── corrected_mlt_Latn.jsonl      # Final Maltese benchmark instances
│   ├── apply_corrections_stats.json  # Re-scoring statistics across languages
│   └── correction_summary.json       # Summary log of corrections and rollbacks
│
├── eval_results/                     # Metric evaluation logs and comparison reports
│   ├── comparison_report.md          # Before vs. After correction metric breakdown
│   ├── eval_before_correction.json   # Full raw evaluation metrics (pre-correction)
│   ├── eval_before_correction_summary.json
│   ├── eval_after_correction.json    # Full raw evaluation metrics (post-correction)
│   └── eval_after_correction_summary.json
│
├── benchmarking/                     # Experimental evaluation suite (75,600 trials)
│   ├── benchmark_runner.py           # Core local / vLLM 6-condition benchmark runner
│   ├── run_api_benchmark_batch.py    # OpenRouter batch API benchmark harness
│   ├── run_api_direct.py             # Direct API execution runner
│   ├── judge_reasons.py              # LLM-as-a-judge reasoning trace evaluator
│   ├── recover_benchmark_nas.py      # Automated Zero-NA fallback recovery protocol
│   ├── run_benchmark.sh              # Local benchmark execution script
│   ├── run_paid_benchmarks.sh        # Frontier API execution script
│   └── results/                      # Full evaluation traces (JSONL per model/language/condition)
│
├── curate_english.py                 # Source filtering & knowledge triple extraction
├── generate_synthetic.py             # Schema-constrained synthetic generation (Stage 1 & 2)
├── sample_dataset.py                 # Balanced stratified sampler (150 items per reasoning level)
├── translate_pipeline.py             # NMT translation & back-translation pipeline
├── translate_judge.py                # Standalone vLLM translation judge
├── quality_correction.py             # Frontier LLM correction pipeline
├── apply_corrections.py              # Correction applicator & LaBSE re-scorer
└── evaluate_quality.py               # Tier 1 (chrF/BLEU/METEOR), Tier 2 (LaBSE/BGE), Tier 3 (COMET/BERTScore)
```

---

## 🛠️ Environment Setup & Installation

```bash
# Clone the repository
git clone https://github.com/mekapilgupta/ikdd-multilingual.git
cd ikdd-multilingual

# Create conda environment
conda create -n multiling_eval python=3.11 -y
conda activate multiling_eval

# Install core dependencies
pip install torch transformers vllm sentence-transformers httpx sacrebleu bert-score unbabel-comet statsmodels scipy
```

---

## 🔬 Reproduction & Pipeline Execution

### 1. English Anchor Curation & Synthetic Data Generation
```bash
# Curate anchor items and extract canonical (s, p, o) triples
python curate_english.py \
    --model-path <MODEL_PATH> \
    --dataset-file eng_Latn.jsonl \
    --output-file curated_eng_Latn.jsonl

# Generate synthetic reasoning questions (Stage 1 Generation + Stage 2 Blind Solver)
python generate_synthetic.py \
    --model-path <MODEL_PATH> \
    --curated-file curated_eng_Latn.jsonl \
    --output-file synthetic_data_v2.jsonl
```

### 2. Balanced Sampling & Multilingual Localization
```bash
# Assemble the 900-item core benchmark (150 items per level L0-L5)
python sample_dataset.py \
    --curated-file curated_eng_Latn.jsonl \
    --synthetic-file synthetic_data_v2.jsonl \
    --output-file sampled_eng_Latn.jsonl \
    --per-level 150

# Run NMT translation with round-trip back-translation
python translate_pipeline.py \
    --sampled-file sampled_eng_Latn.jsonl \
    --output-dir translated

# Execute frontier LLM quality correction
export OPENROUTER_API_KEY="your-api-key"
python quality_correction.py \
    --translated-dir translated \
    --output-dir corrected

# Apply verified corrections & re-score with LaBSE
python apply_corrections.py \
    --translated-dir translated \
    --cache-file correction_cache.jsonl \
    --output-dir corrected
```

### 3. Running the 75,600 Experimental Trials

```bash
# Evaluate local open-weight models (e.g., Qwen)
bash benchmarking/run_benchmark.sh

# Evaluate hosted frontier model snapshots via Batch API (Gemini 3.7 Flash, GPT-5.6 Luna)
export OPENROUTER_API_KEY="your-api-key"
bash benchmarking/run_paid_benchmarks.sh
```

---

## 📈 Summary of Experimental Findings

### 1. Evidence Grounding Contrast ($C_2$ vs. $C_1$)
Providing native continuous passages ($C_2$) increases accuracy by **$+12.46\text{ pp}$** (unadjusted contrast) over closed-book parametric answering ($C_1$), raising mean performance from $82.23\%$ to $94.69\%$ ($\text{OR} = 3.96, p < 0.0001$ under item-clustered logistic regression).

### 2. Paired TOST Equivalence Analysis ($C_5$ vs. $C_4$)
Under Two One-Sided Tests (TOST) with equivalence margin $\delta = \pm 1.5\%$:
- **Gemini 3.7 Flash**: $\Delta = -0.33\text{ pp}$ ($95\%\text{ CI}: [-0.84, +0.18]$, $p_{\text{TOST}} < 0.0001$) $\to$ **Equivalent**.
- **GPT-5.6 Luna**: $\Delta = -0.25\text{ pp}$ ($95\%\text{ CI}: [-0.91, +0.41]$, $p_{\text{TOST}} = 0.0001$) $\to$ **Equivalent**.
- **Qwen 3.7 Flash**: $\Delta = -1.94\text{ pp}$ ($95\%\text{ CI}: [-2.75, -1.14]$, $p_{\text{TOST}} = 0.8593$) $\to$ **Inconclusive**.

### 3. Language Invariance Omnibus Test
An omnibus Wald test across all target languages confirmed that target-language main effects were not statistically significant ($\chi^2(3) = 0.80, p = 0.8495$), with no significant aggregate target-language main effect in this design.

---

## 📄 License

This repository is provided for research use.
- **Code**: Licensed under [Apache 2.0](LICENSE).
- **Data**: Released under [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/).
