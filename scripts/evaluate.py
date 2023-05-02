"""
Evaluates a model on the benchmark dataset.

Usage:
    python evaluate.py --input_file <path to input file> --output_file <path to output file> --model <model name> --mode <baseline or few-shot>

Example:
    python evaluate.py --input_file data/benchmark_samples/hendrycks_sample_0421.csv --output_file data/model_output/results_0421.csv --model text-davinci-003 --mode baseline

Author: Peter Zhang
"""

import argparse
import datetime as dt
from functools import partial

import pandas as pd
from tqdm import tqdm

tqdm.pandas()

from ask_question import ask_row, df_to_examples

INPUT_FILE = f"data/benchmark_samples/hendrycks_sample_{dt.date.today().strftime('%m%d')}.csv"
OUTPUT_FILE = f"data/model_output/results_{dt.date.today().strftime('%m%d')}.csv"

def evaluate_metrics(
        df: pd.DataFrame,
        model: str="text-davinci-003",
        mode: str="baseline",
        ):
    """
    Asks a set of questions and evaluates accuracy and calibration.
    
    Args:
        df: a dataframe with the following columns:
            question: the question text
            choice_A: the first choice
            choice_B: the second choice
            choice_C: the third choice
            choice_D: the fourth choice
            correct_answer: the correct answer
            topic: the topic of the question
            split: the split of the question
        model: the model to use for answering the questions

    Returns:
        A dataframe with the same columns as the input
        dataframe, plus the following columns:
            answer: the model"s answer
            logprobs: the log probabilities of the model"s answer
            justification: the model"s justification
    """

    if mode == "few-shot":
        dev_df = df[df["split"]=="dev"]
        examples = df_to_examples(dev_df, n_examples=3)
        df = df[df["split"]=="test"]
        result = df.progress_apply(partial(ask_row, model=model, mode="few-shot",examples=examples), axis=1)

    else:
        result = df.progress_apply(partial(ask_row, model=model), axis=1)

    df[["answer", "logprobs", "justification"]] = pd.DataFrame(
        result.tolist(), index=df.index)

    return df

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default=INPUT_FILE)
    parser.add_argument("--output_file", type=str, default=OUTPUT_FILE)
    parser.add_argument("--model", type=str, default="text-davinci-003")
    parser.add_argument("--mode", type=str, default="baseline")

    args = parser.parse_args()

    # load test data
    df = pd.read_csv(args.input_file)

    # evaluate
    df = evaluate_metrics(df, model=args.model, mode=args.mode)

    # save results
    df.to_csv(args.output_file, index=False)

if __name__=="__main__":
    main()