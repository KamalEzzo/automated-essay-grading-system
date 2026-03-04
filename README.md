# Automated Essay Grading with Fine-Tuned Gemma 2 9B-IT

This repository contains the complete code, data, and analysis for an automated short-answer grading system built on fine-tuned Gemma 2 9B-IT using LoRA. The system evaluates student responses to accounting questions using a 4-criterion rubric (Clarity, Terminology, Coverage, Accuracy) and produces grades on a 0–5 scale.

## Dataset

- **2,550 student responses** across 85 questions (30 responses each)
- 4 rubric criteria: Clarity (0–2), Terminology (0–2), Coverage (0–2), Accuracy (0–4)
- Final grade = sum of rubric scores / 2 → 0–5 scale
- Question-level stratified split: 51 train / 17 validation / 17 test questions (60/20/20%)

## Key Results

| Model | ±1.0 Accuracy | MAE | QWK | Off-Topic Acc |
|---|---|---|---|---|
| **Fine-tuned Gemma 2 9B** | **78.4%** | **0.713** | **0.821** | **93.1%** |
| Zero-shot Gemma 2 9B | 72.4% | 0.875 | 0.761 | 77.3% |
| Claude Opus 4.6 | 77.1% | 0.749 | 0.813 | 91.8% |
| Claude Sonnet 4 | 77.5% | 0.813 | 0.787 | 84.1% |
| GPT-4o | 71.6% | 0.845 | 0.766 | 88.4% |
| GPT-5.2 | 68.0% | 0.897 | 0.771 | 91.8% |

## Repository Structure

```
├── Notebooks/                          # Jupyter notebooks (main entry point)
│   ├── 01_Data_Validation.ipynb        # Instrument validation (KMO, alpha, DIF, MLM)
│   ├── 02_Zero_Shot_Inference.ipynb    # Run 3 models zero-shot (Colab GPU)
│   ├── 03_Zero_Shot_Analysis.ipynb     # Zero-shot results analysis
│   ├── 04_Data_Splitting.ipynb         # Stratified split algorithm
│   ├── 05_Ablation_Study.ipynb         # LoRA hyperparameter tuning (13 configs)
│   ├── 06_Finetuning_Model.ipynb       # Fine-tuning on Colab (GPU)
│   ├── 07_Finetuning_Results.ipynb     # Fine-tuning evaluation & statistical tests
│   ├── 08_Commercial_Models_Inference.ipynb  # Run GPT & Claude models (API)
│   ├── 09_Commercial_Comparison.ipynb  # Commercial model comparison analysis
│   └── 10_Error_Analysis.ipynb         # Error categorization & robustness
│
├── Data Statsitical Analysis/          # Dataset & validation results
│   ├── Data.xlsx                       # Full dataset (2,550 responses)
│   └── *.csv, *.json                   # Pre-computed validation results
│
├── Zeroshot/                           # Zero-shot inference results
│   ├── results_Gemma2-9B-IT_full_dataset.csv
│   ├── results_Llama3-8B-Instruct_full_dataset.csv
│   └── results_DeepSeek-7B-Chat_full_dataset.csv
│
├── Finetuning/                         # Fine-tuning data & results
│   ├── finetuned_results.csv           # Fine-tuned model test results (510 samples)
│   └── zeroshot_finetuned_prompt_results.csv  # Zero-shot baseline with fine-tuned prompt (510 samples)
│
├── Ablation Study/                     # 13 LoRA configurations evaluated on validation set
│   ├── r8/, r16/, ..., r128/          # LoRA rank sweep
│   ├── lr1e-06/, ..., lr2e-05/        # Learning rate sweep
│   ├── r32_5epochs/, r80_dropout0.1/  # Extended training & dropout
│   └── summary_*.csv                  # Aggregated sweep summaries
│
├── GPT_Comparison/                     # GPT-4o & GPT-5.2 results
│   ├── gpt-4o/                        # GPT-4o test results
│   └── gpt-5_2/                       # GPT-5.2 test results
│
├── Claude_Comparison/                  # Claude Opus & Sonnet results
│   ├── claude_opus_4_6/               # Claude Opus 4.6 test results
│   └── claude_sonnet_4_20250514/      # Claude Sonnet 4 test results
│
├── requirements.txt
└── .gitignore
```

## Notebooks Guide

The notebooks are numbered in thesis pipeline order:

| # | Notebook | Runs On | Description |
|---|---|---|---|
| 01 | Data Validation | Local CPU | Instrument validation: KMO (0.857), Cronbach's alpha (0.927), DIF analysis, multilevel modeling |
| 02 | Zero-Shot Inference | Colab GPU | Runs Gemma 2 9B, Llama 3 8B, DeepSeek 7B on full dataset (2,550 samples) |
| 03 | Zero-Shot Analysis | Local CPU | Compares 3 zero-shot models, selects Gemma for fine-tuning |
| 04 | Data Splitting | Local CPU | Implements stratified question-level split with 50K-iteration optimization |
| 05 | Ablation Study | Local CPU | Analyzes 13 LoRA configurations with bootstrap significance tests |
| 06 | Fine-tuning Model | Colab GPU | Fine-tunes Gemma 2 9B-IT with LoRA (r=80, LR=2e-5, 3 epochs) |
| 07 | Fine-tuning Results | Local CPU | Before/after comparison with statistical tests (paired t-test, Wilcoxon, TOST equivalence) |
| 08 | Commercial Inference | Local CPU | Runs GPT-4o, GPT-5.2, Claude Opus 4.6, Claude Sonnet 4 via API |
| 09 | Commercial Comparison | Local CPU | Compares all 6 models with paired bootstrap significance tests |
| 10 | Error Analysis | Local CPU | Categorizes grading errors into 7 types, identifies per-question weaknesses |

## Getting Started

### Analysis notebooks (01, 03–05, 07, 09–10)

These load pre-computed results and run locally — no GPU or API keys needed:

```bash
pip install -r requirements.txt
jupyter notebook Notebooks/
```

### Inference notebooks (02, 06)

These require a Google Colab GPU runtime (T4 or better):
- Upload the notebook to Google Colab
- Mount Google Drive with the dataset
- Run all cells

### Commercial model inference (08)

Requires API keys:

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
jupyter notebook Notebooks/08_Commercial_Models_Inference.ipynb
```

## Evaluation Metrics

- **±1.0 Accuracy**: Percentage of predictions within 1.0 point of human grade (0–5 scale)
- **±0.5 Accuracy**: Stricter threshold
- **MAE**: Mean Absolute Error
- **QWK**: Quadratic Weighted Kappa (inter-rater agreement)
- **Off-Topic Accuracy**: Correct classification of on-topic, off-topic, and no-answer responses
