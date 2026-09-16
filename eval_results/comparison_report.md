# Translation Quality: Before vs After Frontier-LLM Correction

**Generated**: 2026-08-19 03:51:30

**Stage 1 (Gemini)**: google/gemini-3.7-flash:batch

**Stage 2 (Qwen)**:   qwen/qwen3-7-flash

---


## Odia (`ory_Orya`)

| Metric | Before | After | Δ | Direction |
|--------|--------|-------|---|-----------|
| **CHRF** | 3.2537 | 3.1205 | -0.1332 | ❌ ↓ |
| **BLEU** | 0.0147 | 0.0161 | +0.0014 | ➡ = |
| **METEOR** | 0.6910 | 0.4966 | -0.1944 | ❌ ↓ |
| **LABSE** | 0.8756 | 0.8907 | +0.0151 | ✅ ↑ |
| **BGE_M3** | 0.7966 | 0.8026 | +0.0060 | ➡ = |
| **QWEN_EMBEDDING** | 0.6619 | 0.6710 | +0.0091 | ➡ = |

## Assamese (`asm_Beng`)

| Metric | Before | After | Δ | Direction |
|--------|--------|-------|---|-----------|
| **CHRF** | 3.1518 | 3.2261 | +0.0743 | ✅ ↑ |
| **BLEU** | 0.0769 | 0.0759 | -0.0010 | ➡ = |
| **METEOR** | 0.5339 | 0.3787 | -0.1552 | ❌ ↓ |
| **LABSE** | 0.7679 | 0.7758 | +0.0079 | ➡ = |
| **BGE_M3** | 0.7683 | 0.7745 | +0.0062 | ➡ = |
| **QWEN_EMBEDDING** | 0.6339 | 0.6414 | +0.0075 | ➡ = |

## Sinhala (`sin_Sinh`)

| Metric | Before | After | Δ | Direction |
|--------|--------|-------|---|-----------|
| **CHRF** | 3.9499 | 4.3825 | +0.4326 | ✅ ↑ |
| **BLEU** | 0.0700 | 0.0666 | -0.0034 | ➡ = |
| **METEOR** | 1.4798 | 1.4708 | -0.0090 | ➡ = |
| **LABSE** | 0.8836 | 0.9035 | +0.0199 | ✅ ↑ |
| **BGE_M3** | 0.7961 | 0.8075 | +0.0114 | ✅ ↑ |
| **QWEN_EMBEDDING** | 0.4342 | 0.4718 | +0.0376 | ✅ ↑ |

## Maltese (`mlt_Latn`)

| Metric | Before | After | Δ | Direction |
|--------|--------|-------|---|-----------|
| **CHRF** | 25.7856 | 27.0929 | +1.3073 | ✅ ↑ |
| **BLEU** | 0.2953 | 0.3352 | +0.0399 | ✅ ↑ |
| **METEOR** | 6.1616 | 5.9938 | -0.1678 | ❌ ↓ |
| **LABSE** | 0.8976 | 0.9287 | +0.0311 | ✅ ↑ |
| **BGE_M3** | 0.6983 | 0.7035 | +0.0052 | ➡ = |
| **QWEN_EMBEDDING** | 0.7084 | 0.7199 | +0.0115 | ✅ ↑ |

---

## Token Usage (API Cost)

| Stage | Model | Items | Prompt Tokens | Completion Tokens | Total |
|-------|-------|-------|---------------|-------------------|-------|
| stage1_gemini | `google/gemini-3.7-flash:batch` | 1,000 | 802,020 | 1,164,113 | 1,966,133 |
| stage1_gemini | `google/gemini-3.7-flash:batch` | 13 | 11,715 | 20,118 | 31,833 |
| stage1_gemini | `google/gemini-3.7-flash:batch` | 1,000 | 816,506 | 1,673,893 | 2,490,399 |
| stage1_gemini | `google/gemini-3.7-flash:batch` | 1,000 | 647,214 | 1,277,071 | 1,924,285 |
| stage1_gemini | `google/gemini-3.7-flash:batch` | 1,000 | 684,248 | 1,291,741 | 1,975,989 |
| stage2_qwen | `qwen/qwen3.7-flash` | 548 | 455,106 | 387,257 | 842,363 |

**Total tokens used**: 9,231,002