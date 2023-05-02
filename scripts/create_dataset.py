"""
Creates a dataset from the MMLU dataset.

The MMLU dataset is available at https://github.com/hendrycks/test.

Usage: 
    python create_dataset.py --dataset_folder <path to dataset folder> --splits <splits to include> --topics <topics to include> --output_file <path to output file>

Example:
    python create_dataset.py --dataset_folder data/hendrycks_test --splits dev val --topics arithmetic --output_file data/benchmark_samples/hendrycks_sample_0421.csv

Author: Peter Zhang
"""

import argparse
import datetime as dt
import glob
from os.path import join
from typing import List

import pandas as pd

from config import topics

DATASET_FOLDER = "data/hendrycks_test"
SPLITS = ["dev","val"] # by default, only use the dev and val splits
OUTPUT_FILE = f"data/benchmark_samples/hendrycks_sample_{dt.date.today().strftime('%m%d')}.csv"

RS = 42
QUESTION_COUNT = 35

def build_benchmark(dataset_folder: str, splits: List[str], topics: List[str]) -> pd.DataFrame:
    """
    Assembles a dataframe with all questions from the MMLU dataset.

    Args:
        dataset_folder: path to the dataset folder
        splits: list of splits to include
        topics: list of topics to include

    Returns:
        A dataframe with the following columns:
            question: the question text
            choice_A: the first choice
            choice_B: the second choice
            choice_C: the third choice
            choice_D: the fourth choice
            correct_answer: the correct answer
            topic: the topic of the question
            split: the split of the question
    """

    all_dfs = []
    for split in splits:
        files = glob.glob(join(dataset_folder, split, "*.csv"))
        for topic in topics:
            file = join(dataset_folder, split, f"{topic}_{split}.csv")
            assert file in files, f"Could not find file: {file}"
            df = pd.read_csv(file, header=None)
            df["topic"] = topic
            df["split"] = split
            all_dfs.append(df)


    benchmark_df = pd.concat(all_dfs, axis=0)
    columns = ["question","choice_A","choice_B","choice_C","choice_D","correct_answer"]
    benchmark_df.rename(dict(zip(range(6), columns)), axis=1, inplace=True)

    return benchmark_df

def sample_benchmark(
        benchmark_df: pd.DataFrame,
        num_questions: int=QUESTION_COUNT,
        random_state: int=RS
        ) -> pd.DataFrame:
    """
    Samples a subset of the benchmark dataframe.

    Args:
        benchmark_df: the benchmark dataframe
        num_questions: the number of questions to sample
        random_state: the random state

    Returns:
        A dataframe with the same columns as the input dataframe
    """

    num_val_questions = num_questions - 5 # we will add 5 questions from the dev set

    sample_df = benchmark_df[benchmark_df["split"] != "dev"]
    dev_df = benchmark_df[benchmark_df["split"] == "dev"]
    counts = sample_df["topic"].value_counts()
    good_topics = counts[counts >= num_val_questions].index.tolist() # hacky but preserves original generation
    good_val_df = sample_df[sample_df["topic"].isin(good_topics)].groupby("topic").sample(num_val_questions, random_state=random_state)
    bad_val_df = sample_df[~sample_df["topic"].isin(good_topics)]
    benchmark_df = pd.concat([good_val_df, bad_val_df, dev_df], axis=0)

    return benchmark_df

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_folder", type=str, default=DATASET_FOLDER)
    parser.add_argument("--topics", type=str, nargs="+", default=topics)
    parser.add_argument("--splits", type=str, nargs="+", default=SPLITS)
    parser.add_argument("--num_questions", type=int, default=QUESTION_COUNT)
    parser.add_argument("--random_state", type=int, default=RS)
    parser.add_argument("--output_file", type=str, default=OUTPUT_FILE)
    args = parser.parse_args()

    benchmark_df = build_benchmark(args.dataset_folder, args.splits, args.topics)
    benchmark_df = sample_benchmark(benchmark_df, args.num_questions, args.random_state)
    benchmark_df.to_csv(args.output_file, index=False)
    
if __name__ == "__main__":
    main()