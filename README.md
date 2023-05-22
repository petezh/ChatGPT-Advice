Taking Advice from ChatGPT
---
Last Updated on 05/20/2023

Author: Peter Zhang

[Paper link](https://psyarxiv.com/b53vn)

This study treats GPT models as an advisor in a [judge-advisor system](https://www.sciencedirect.com/science/article/abs/pii/S0749597800929261). We prompt InstructGPT using CoT on the [MMLU](https://arxiv.org/abs/2009.03300) benchmark and treat model output as "advice." In our lab study, 118 student participants answer 2,828 questions and recieve a chance to update their answer from the advice. We analyze factors affecting weight on advice and participant confidence in advice answers. This repository contains all of the collected data as well as code to reproduce the tables and figures in the paper.

![](figures/study_design.png)

- [Setup](#setup)
- [Data](#data)
  - [Model Output](#model-output)
  - [Survey Responses](#survey-responses)
- [Scripts](#scripts)
- [Notebooks](#notebooks)
- [Outputs](#outputs)

# Setup
1. Install packages using the following:
```
pip install -r requirements.txt
```

2. Add your OpenAI API key to [`ask_question.py`](scripts/ask_question.py).
3. Download the MMLU benchmark from the [official repository](https://github.com/hendrycks/test) and move it to [`hendrycks_test`](data/hendrycks_test/) 

# Data

## Model Output
- Relevant samples from the benchmark are in the [`benchmark_samples`](data/benchmark_samples/) folder.
- The output used to as advice in the survey is [`results_scratchpad_0218.csv`](data/model_output/results_scratchpad_0218.csv) which is an evaluation on [`hendrycks_sample_0217.csv`](data/benchmark_samples/hendrycks_sample_0217.csv).
- The output used in the final evaluation is [`results_0425.csv.csv`](data/model_output/results_0425.csv) which is evaluated on [`hendrycks_sample_0423.csv`](data/benchmark_samples/hendrycks_sample_0423.csv).

## Survey Responses
- The survey responses are downloaded from Qualtrics and placed in the [`survey_responses`](data/survey_responses/) folder.
- The updated survey responses from April 23rd are stored in [`responses_0425.csv](data/survey_responses/responses_0425.csv).

# Scripts

The [`create_dataset.py`](scripts/create_dataset.py) script creates a sample of the MMLU benchmark. The [`evaluate.py`](scripts/evaluate.py) script supports multiple types of prompting. The templates used in the evaluation script are located in the [`templates`](templates) folder.

To reproduce the model evaluation, run [`cot_eval.sh`](scripts/cot_eval.sh) from the main repository folder.

The [`clean_responses.py`](scripts/clean_responses.py) script preprocesses the survey responses and creates features used in the analysis.

# Notebooks

- The [`analyze_model_results.ipynb`](notebooks/analyze_model_results.ipynb) notebook produces figures displaying LLM performance.
- The [`explore_questions.ipynb](notebooks/explore_questions.ipynb) notebook creates some summary statistics about benchmark questions.
- The [`analyze_survey_responses.ipynb`](notebooks/analyze_survey_responses.ipynb) notebook produces all other figures and tables.
- To create survey questions from benchmark samples, use [`generate_survey_questions.ipynb`](notebooks/generate_survey_questions.ipynb).

# Outputs

All [`figures`](figures) and [`tables`](tables) are output to their respective folders.
