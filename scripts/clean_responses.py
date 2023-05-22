"""
Cleans the Qualtrics responses.

Usage:
    python clean_responses.py --output_folder <path to output folder> --responses_folder <path to responses folder> --responses_file <name of responses file> --min_time <minimum time to consider a response> --min_score <minimum score to consider a response> --target_phases <list of target phases>

Example:
    python clean_responses.py --output_folder data/cleaned_responses --responses_folder data/survey_responses --responses_file hmc-v1_April 25, 2023_12.27.csv --min_time 1000 --min_score -inf --target_phases phase 1 phase 2 phase 3

Author: Peter Zhang
"""

import argparse
from collections import defaultdict
import datetime as dt
import json
import logging
from os.path import join
from typing import Dict, List, Tuple

import pandas as pd
import config

RESPONSES_FOLDER = "data/survey_responses"
OUTPUT_FOLDER = "data/cleaned_responses"
RESPONSES_FILE = "responses_0425.csv"
SURVEY_QUESTIONS_FOLDER = "data/survey_questions"
MIN_TIME = 1000 # minimum seconds to consider a survey response
MIN_SCORE = float("-inf")

USAGE_COLS = ['heard_of', 'used', 'used_in_class', 'answered_mc'] # each level of usage
FAMILIARITY_COLS = [f"familiarity_{i}" for i in range(1, 9)] # familiarity with 8 topics
 # name of topic areas

topic2col = dict(zip(config.fam_topics, FAMILIARITY_COLS))
MAJORS_FILE = "majors_rename.json"
PARTICIPANT_COLS = ["score", "major", "time", "usage_description"] + USAGE_COLS + FAMILIARITY_COLS # list of all participant-level features
TARGET_PHASES = ['phase 1', 'phase 2', 'phase 3']

PHASE2FP = {
    'phase 1': join(SURVEY_QUESTIONS_FOLDER, "qualtrics_0413.json"),
    'phase 2': join(SURVEY_QUESTIONS_FOLDER, "qualtrics_0417.json"),
    'phase 3': join(SURVEY_QUESTIONS_FOLDER, "qualtrics_0421.json"),
}


"""
=====================
PREPROCESSING SCRIPTS
=====================
"""


def clean_qualtrics(qualtrics: dict) -> dict:
    """Extract answers from qualtrics json file."""

    qualtrics = {int(q): qualtrics[q] for q in qualtrics}
    for k, v in qualtrics.items():
        question, optionA, optionB, optionC, optionD, advice_answer, justification, correct_answer, topic = v
        qualtrics[k] = {
            'question': question,
            'optionA': optionA,
            'optionB': optionB,
            'optionC': optionC,
            'optionD': optionD,
            'correct_answer': correct_answer,
            'justification': justification,
            'advice_answer': advice_answer,
            'topic': topic,
        }
    return qualtrics

load_json = lambda fp: json.loads(open(fp, 'r').read().replace('\\\\','\\'))
pipeline = lambda fp: clean_qualtrics(load_json(fp))
phase2qualtrics = {wave:pipeline(fp) for wave, fp in PHASE2FP.items()}

def preprocess(survey_df: pd.DataFrame,
               min_time: int,
               min_score: float, 
               target_phases: List[str],
               responses_folder: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Preprocess the survey responses."""

    survey_df = preprocess_init_filter(survey_df, min_time, min_score)
    survey_df = preprocess_usage(survey_df)
    survey_df = preprocess_major(survey_df, responses_folder)
    survey_df = preprocess_phase(survey_df, target_phases)
    q2col = get_q2col(survey_df)
    dataset = survey_df.apply(process_questions, axis=1, q2col=q2col)
    dataset = [d for row in dataset for d in row]
    question_df = pd.DataFrame(dataset)
    participant_df = survey_df[PARTICIPANT_COLS]

    logging.info(f"Processed {len(question_df)} questions from {len(survey_df)} responses.")

    return participant_df, question_df

def preprocess_init_filter(survey_df: pd.DataFrame,
                           min_time: int,
                           min_score: float) -> pd.DataFrame:
    """Preprocess the survey responses by filtering out the first two rows,
    filtering by time, and filtering by score."""

    # drop the first two rows
    survey_df = survey_df.iloc[2:]

    before_init_filter = len(survey_df)

    # set filter by time
    survey_df.rename({"Duration (in seconds)":"time"}, inplace=True, axis=1)
    survey_df = survey_df[survey_df['time'].astype(int) > min_time]

    # filter to lab tests
    survey_df = survey_df[survey_df['Status'] == 'IP Address']

    survey_df['score'] = survey_df['score'].astype(float) # turn score to float
    survey_df = survey_df[survey_df['score'] > min_score] # filter by score

    dropped_by_filter = len(survey_df) - before_init_filter
    logging.info(f"Filtered {dropped_by_filter} responses based on time and score.")

    return survey_df

def preprocess_usage(survey_df: pd.DataFrame) -> pd.DataFrame:
    """Process the usage columns of the survey."""

    for col in USAGE_COLS:
        survey_df[col] = survey_df[col] == 'Yes' # encode as boolean
    return survey_df

def preprocess_major(survey_df: pd.DataFrame, responses_folder: str) -> pd.DataFrame:
    """Process the major columns of the survey."""
    
    majors_dict = defaultdict(lambda: "Other")
    majors_dict.update(json.load(open(join(responses_folder, MAJORS_FILE), 'r')))
    rename_majors = lambda orig_major: majors_dict[orig_major]
    survey_df['major'] = survey_df['major'].apply(rename_majors)

    return survey_df

def date2phase(date: dt.date) -> str:
    """Classify survey phase."""

    if date < dt.date(2023, 4, 4):
        return 'tests'
    if date <  dt.date(2023, 4, 13):
        return 'pilots'
    elif date == dt.date(2023, 4, 13):
        return 'phase 1'
    elif (date >= dt.date(2023, 4, 17)) and (date < dt.date(2023, 4, 21)):
        return 'phase 2'
    elif date >= dt.date(2023, 4, 21):
        return 'phase 3'
    return 'drop'

def preprocess_phase(survey_df: pd.DataFrame, target_phases: List[str]) -> pd.DataFrame:
    """Process the phase columns of the survey."""
    
    survey_df['date'] = pd.to_datetime(survey_df['RecordedDate']).dt.date
    survey_df['phase'] = survey_df['date'].apply(date2phase)
    survey_df = survey_df[survey_df['phase'].isin(target_phases)] # only consider responses from the 3 phases
    survey_df = survey_df[survey_df['source'] == 'rpp'] # restrict to rpp survey respondents

    return survey_df

def get_q2col(survey_df: pd.DataFrame) -> dict:
    """Get mappers from column names to question numbers."""

    columns = survey_df.columns
    column_numbers = [int(c.split('_')[0]) for c in columns[columns.str.contains('adjustment_1')]]  # get names of all columns
    q2col = dict(zip(range(1, len(column_numbers)+1), column_numbers))
    return q2col

process_qlist = lambda qlist: [int(qid) for qid in qlist.split(',') if qid != '']

def process_questions(row: pd.Series, q2col: dict) -> List[dict]:
    """Process the questions for a given row."""
    
    # get the questions list
    questions_list = process_qlist(row['questions_list'])

    # assemble participant-level data
    participant_data = {
        'participant_id':row['ResponseId'],
        'source': row['source'],
        'advisor': row['advisor'],
        'give_justification': row['justification'],
    }
    for col in PARTICIPANT_COLS:
        participant_data[col] = row[col]

    row_dataset = []

    for i, question_id in enumerate(questions_list):

        columnnum = q2col[question_id]
        question_data = {}
        
        for stage, colname in [("init","question"), ("adjusted", "adjustment")]:
            total = sum([float(row[f"{columnnum}_{colname}_{choice_num}"])
                         for choice_num in [1, 2, 3, 4]])
            time = float(row[f"{columnnum}_{colname}_timer_Page Submit"])
            for choice_num, letter in zip([1, 2, 3, 4], "ABCD"):
                question_data[f"{stage}_choice{letter}"] = float(row[f"{columnnum}_{colname}_{choice_num}"]) / total
            question_data[f"{stage}_time"] = time
        
        question_data["question_num"] = i+1
        qualtrics = phase2qualtrics[row["phase"]]
        question_meta = qualtrics[question_id]

        question_data.update(question_meta)
        question_data.update(participant_data)
        row_dataset.append(question_data)
    
    return row_dataset


"""
===========================
FEATURE ENGINEERING SCRIPTS
===========================
"""


def feature_engineer(question_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create permanent features.
    """

    question_df = add_weight_on_advice(question_df)
    question_df = add_topics(question_df)
    question_df[USAGE_COLS] = question_df[USAGE_COLS].astype(int)
    question_df["usage_level"] = question_df[USAGE_COLS].sum(axis=1)
    question_df["question_group"] = pd.cut(question_df["question_num"], bins=range(0, 41, 5))
    question_df["question_id"] = question_df.apply(hash_question, axis=1)
    question_df = add_advice_confidence(question_df)

    return question_df

def add_weight_on_advice(question_df: pd.DataFrame) -> pd.DataFrame():
    """Calculate weight on advice and winsorize."""

    question_df['weight_on_advice'] = question_df.apply(calc_woa, axis=1)
    neg_woa_count = sum(question_df['weight_on_advice'] < 0)
    logging.info(f"Winsorizing {neg_woa_count} questions with negative weight.")
    question_df['weight_on_advice'] = question_df['weight_on_advice'].clip(lower=0)

    return question_df

def calc_woa(row: pd.Series) -> float:
    """Calculate weight on advice following Logg et al. (2019)."""
    
    if row['advice_answer'] not in 'ABCD':
        return 0
    elif 1-row[f'init_choice{row["advice_answer"]}'] == 0:
        return 0
    else:
        return (row[f'adjusted_choice{row["advice_answer"]}'] - \
            row[f'init_choice{row["advice_answer"]}'])/(1-row[f'init_choice{row["advice_answer"]}'])

def get_topic_familiarity(row: pd.Series) -> str:
    """Gets the familiarity level of the participant for the familiarity topic category."""
    
    col = topic2col[row["fam_topic"]]

    return row[col]

def add_topics(question_df: pd.DataFrame) -> pd.DataFrame():
    """Calculates net familiarity."""

    question_df['fam_topic'] = question_df['topic'].apply(config.topic2fam_topic.get)

    question_df['topic_familiarity'] = question_df.apply(get_topic_familiarity, axis=1) 
    question_df['net_familiarity'] = question_df['topic_familiarity'].apply(config.comfort_map.get)

    for comfort_level in ("Uncomfortable", "Neutral", "Comfortable"):
        question_df[comfort_level.lower()] = (question_df["topic_familiarity"] == comfort_level).astype(int)

    question_df = question_df.drop(FAMILIARITY_COLS, axis=1)

    return question_df

def hash_question(row: pd.Series) -> str:
    """Creates a hash from the question text and choice text."""

    text = row["question"]
    for choice in "ABCD":
        text += row[f"option{choice}"]
    return hash(text)

def add_advice_confidence(question_df: pd.DataFrame) -> pd.DataFrame:
    """Add features for participant confidence in advice."""

    # whether advice is correct
    question_df['advice_is_correct'] = (question_df['advice_answer'] == question_df['correct_answer']).astype(int)
    question_df['last_advice_is_correct'] = question_df.groupby('participant_id')['advice_is_correct'].shift(1).fillna(0)

    # previous advice performance
    question_df['correct_advice_count'] = question_df.groupby('participant_id', group_keys=False)['advice_is_correct'].apply(lambda x: x.shift(1).cumsum().fillna(0))
    question_df['incorrect_advice_count'] = question_df.groupby('participant_id', group_keys=False)['advice_is_correct'].cumcount() - question_df['correct_advice_count']

    # initial and adjusted belief in advice answer
    init_advice_confidence = lambda row: row[f"init_choice{row['advice_answer']}"] if row['advice_answer'] in "ABCD" else 0
    question_df['init_advice_confidence'] = question_df.apply(init_advice_confidence, axis=1)
    get_advice_prob = lambda row: row[f"adjusted_choice{row['advice_answer']}"] if row['advice_answer'] in "ABCD" else 0
    question_df['advice_confidence'] = question_df.apply(get_advice_prob, axis=1)
    
    return question_df

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_folder", type=str, default=OUTPUT_FOLDER)
    parser.add_argument("--responses_folder", type=str, default=RESPONSES_FOLDER)
    parser.add_argument("--responses_file", type=str, default=RESPONSES_FILE)
    parser.add_argument("--min_time", type=int, default=MIN_TIME)
    parser.add_argument("--min_score", type=float, default=MIN_SCORE)
    parser.add_argument("--target_phases", type=str, nargs='+', default=TARGET_PHASES)
    
    args = parser.parse_args()
    
    survey_df = pd.read_csv(join(args.responses_folder, args.responses_file), low_memory=False)
    participant_df, question_df = preprocess(
        survey_df,
        args.min_time,
        args.min_score,
        args.target_phases,
        args.responses_folder,
    )

    output_file = join(args.output_folder, args.responses_file.replace('.csv', '_cleaned.csv'))
    question_df.to_csv(output_file, index=False)

    output_file = join(args.output_folder, args.responses_file.replace('.csv', '_responses.csv'))
    participant_df.to_csv(output_file, index=False)
    
    question_df = feature_engineer(question_df)

    output_file = join(args.output_folder, args.responses_file.replace('.csv', '_features.csv'))
    question_df.to_csv(output_file, index=False)

if __name__=="__main__":
    main()