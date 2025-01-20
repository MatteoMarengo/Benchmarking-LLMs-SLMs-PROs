# Benchmarking Small Language Models for Radiotherapy Follow-Up

This repository contains code to **benchmark multiple small Large Language Models (LLMs)** for summarizing patient responses to radiotherapy follow-up questionnaires. We compare models such as **Mistral-7B Instruct**, **BioMistral-7B**, **LLaMA-2-7B-Chat**, **Gemma-7B Instruct**, and **GPT-4** (as a reference).

This repository is the official code for the paper [Benchmarking LLMs and SLMs for patient reported outcomes](https://arxiv.org/abs/2412.16291) written by Matteo MARENGO, Jarod LEVY and Jean-Emmanuel BIBAULT.

## Table of Contents
- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
  - [1. Conda Installation](#1-conda-installation)
  - [2. Pip Installation](#2-pip-installation)
- [Usage](#usage)
  - [Environment Variables](#environment-variables)
  - [Running Benchmarks](#running-benchmarks)
- [Details of the Scripts](#details-of-the-scripts)
  - [Utility Modules](#utility-modules)
  - [Benchmark Scripts](#benchmark-scripts)
- [Hugging Face and OpenAI Tokens](#hugging-face-and-openai-tokens)
- [Results](#results)
- [License](#license)
- [Citation](#citation)

---

## Project Overview

### Objective
The primary goal of this project is to **evaluate how well different language models can summarize** patient Q&A data from radiotherapy follow-up questionnaires. Each questionnaire contains items regarding:
- Number of radiation treatments
- Symptom severity (fatigue, pain, diarrhea, etc.)
- Frequency of daily-activity interference
- Additional open-ended responses

We assess these models with:
1. **Custom Metric** (coverage of key symptoms if severity > 0.5)  
2. **Recall** (how many relevant symptoms are captured)  
3. **Cohen’s Kappa** (compares summarized results vs. ground truth, correcting for random chance)  
4. **LLM-Based Score** (GPT-4's rating of summary accuracy)

---

## Repository Structure

```plaintext
benchmark-small-models/
├── README.md                 # This README
├── LICENSE                   # (Optional) e.g. MIT
├── .env.example              # Example environment variables
├── requirements.txt          # Dependencies for pip
├── environment.yml           # Dependencies for conda
│
├── src/
│   ├── __init__.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── scoring_utils.py
│   │   ├── prompt_utils.py
│   │   ├── evaluation_utils.py
│   │   └── plotting_utils.py
│   └── benchmarks/
│       ├── __init__.py
│       ├── benchmark_mistral7b_instruct.py
│       ├── benchmark_biomistral7b.py
│       ├── benchmark_llama2_7b_chat.py
│       ├── benchmark_gemma7b_instruct.py
│       └── benchmark_gpt4.py
│
│
└── saved_files/
    ├── Benchmark_MISTRAL7B_INSTRUCT/
    │   ├── prompt/
    │   ├── output/
    │   └── figures/
    ├── Benchmark_BIOMISTRAL7B/
    │   ├── prompt/
    │   ├── output/
    │   └── figures/
    ├── Benchmark_LLAMA27B/
    │   ├── prompt/
    │   ├── output/
    │   └── figures/
    ├── Benchmark_GEMMA7B_INSTRUCT/
    │   ├── prompt/
    │   ├── output/
    │   └── figures/
    └── Benchmark_GPT-4/
        ├── prompt/
        ├── output/
        └── figures/
```

## Installation

We recommend using either conda or pip to set up the environment.

### 1. Conda Installation
#### 1. Clone the repository:
```bash
git clone https://github.com/YourUsername/benchmark-small-models.git
cd benchmark-small-models
```
#### 2. Create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate benchmark-llm
```
#### 3. Verify the installation:
```bash
python --version
python -m pip list
```

### 2. Pip Installation

#### 1. Clone the repository:
```bash
git clone https://github.com/YourUsername/benchmark-small-models.git
cd benchmark-small-models
```

#### 2. Install dependencies via pip:
```bash
pip install -r requirements.txt
```

#### 3. Verify the installation:
```bash
python --version
python -m pip list
```

## Usage
### Environment Variables
#### 1. Copy .env.example to .env:
```bash
cp .env.example .env
```

#### 2. Edit .env to include your tokens:
```bash
HUGGINGFACE_TOKEN=hf_XXXX
OPENAI_API_KEY=sk-XXXX
```

### Running Benchmarks

Each benchmark script is under src/benchmarks/. For example, to run Mistral-7B Instruct:
```bash
python src/benchmarks/benchmark_mistral7b_instruct.py
```

Likewise, for other models:

- benchmark_biomistral7b.py
- benchmark_llama2_7b_chat.py
- benchmark_gemma7b_instruct.py
- benchmark_gpt4.py (requires valid OpenAI API key)

Each script will:

  - Generate X random prompts and associated patient responses.
  - Summarize them using the respective model.
  - Compute and log metrics: coverage (custom metric), recall, Kappa, GPT-based scoring.
  - Save outputs in saved_files/.

## Details of the Scripts
### Utility Modules

- src/utils/scoring_utils.py
  Maps text answers (e.g., "Mild", "Severe") to numeric values and processes raw Q&A answers.

- src/utils/prompt_utils.py
  Generates random question/answer sets for the radiotherapy questionnaire.

- src/utils/evaluation_utils.py
  - compute_metric(): Measures how many major symptoms appear in the summary.
  - kappa_cohen_index(): Calculates Cohen’s Kappa.
  - recall(): Measures coverage of relevant symptoms.
  - LLM_evaluator(): Calls GPT-4 to obtain a 0–1 score.

- src/utils/plotting_utils.py
    Creates and saves plots of the four metrics (score, kappa, recall, LLM-based grade).

### Benchmark Scripts

Under src/benchmarks/, each file:

  1. Imports the target model from Hugging Face or OpenAI.
  2. Loops over X prompts, calls the model, and scores the output.
  3. Saves results (CSV, text, plots) into its dedicated folder in saved_files/.

## Hugging Face and OpenAI Tokens

  - Store your Hugging Face token in .env if your model is private or large.
  - For GPT-4 usage, you must provide a valid OPENAI_API_KEY.

## Results

Each run produces:

A CSV file (benchmark.csv) with columns:
- Prompt number
- Grade Score (our custom coverage)
- Kappa Score
- Recall Score
- LLM Based Score

- prompt_ files in /prompt/
- output_ files in /output/
- figures (.png files) showing metric evolutions over multiple prompts.


## Citation

If you use this repository for research, please cite or reference it by citing the associated paper:

```bibtex
@misc{marengo2024benchmarkingllmsslmspatient,
  title        = {Benchmarking LLMs and SLMs for patient reported outcomes},
  author       = {Matteo Marengo and Jarod Lévy and Jean-Emmanuel Bibault},
  year         = {2024},
  eprint       = {2412.16291},
  archivePrefix= {arXiv},
  primaryClass = {cs.AI},
  url          = {https://arxiv.org/abs/2412.16291},
}

