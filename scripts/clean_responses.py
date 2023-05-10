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
from os.path import join
from typing import Dict, List, Tuple

import pandas as pd

RESPONSES_FOLDER = "data/survey_responses"
OUTPUT_FOLDER = "data/cleaned_responses"
RESPONSES_FILE = "hmc-v1_April 25, 2023_12.27.csv" 
SURVEY_QUESTIONS_FOLDER = "data/survey_questions"
MIN_TIME = 1000 # minimum seconds to consider a survey response
MIN_SCORE = float("-inf")

USAGE_COLS = ['heard_of', 'used', 'used_in_class', 'answered_mc'] # each level of usage
FAMILIARITY_COLS = [f"familiarity_{i}" for i in range(1, 9)]
FAMILIARITY_TOPICS = [
    'Mathematics',
    'Literature',
    'History',
    'Economics',
    'Biological Sciences',
    'Physics',
    'Computer Science',
    'Trivia',
]
topic2col = dict(zip(FAMILIARITY_TOPICS, FAMILIARITY_COLS))
MAJORS_FILE = "majors_rename.json"
PARTICIPANT_COLS = ['score', 'major', "time"] + USAGE_COLS + FAMILIARITY_COLS # list of all participant-level features
TARGET_PHASES = ['phase 1', 'phase 2', 'phase 3']

PHASE2FP = {
    'phase 1': join(SURVEY_QUESTIONS_FOLDER, "qualtrics_0413.json"),
    'phase 2': join(SURVEY_QUESTIONS_FOLDER, "qualtrics_0417.json"),
    'phase 3': join(SURVEY_QUESTIONS_FOLDER, "qualtrics_0421.json"),
}

def clean_qualtrics(qualtrics):
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
    row2columns, col2q = get_mappers(survey_df)
    dataset = survey_df.apply(process_questions, axis=1, args=(row2columns, col2q))
    dataset = [d for row in dataset for d in row]
    question_df = pd.DataFrame(dataset)

    return survey_df, question_df

def preprocess_init_filter(survey_df: pd.DataFrame,
                           min_time: int,
                           min_score: float) -> pd.DataFrame:
    """Preprocess the survey responses by filtering out the first two rows,
    filtering by time, and filtering by score."""

    # drop the first two rows
    survey_df = survey_df.iloc[2:]

    # set filter by time
    survey_df["time"] = survey_df["Duration (in seconds)"]
    survey_df = survey_df[survey_df['time'].astype(int) > min_time]

    # filter to lab tests
    survey_df = survey_df[survey_df['Status'] == 'IP Address']

    survey_df['score'] = survey_df['score'].astype(float) # turn score to float
    survey_df = survey_df[survey_df['score'] > min_score] # filter by score

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
    rename_majors = lambda orig_major: majors_dict[orig_major].split(',')
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

def get_mappers(survey_df: pd.DataFrame):
    """Get mappers from column names to question numbers."""

    row2columns = {}
    columns = survey_df.columns
    column_numbers = [int(c.split('_')[0]) for c in columns[columns.str.contains('adjustment_1')]]  # get names of all columns
    for i, row in survey_df.iterrows():
        answered = columns[columns.str.contains('adjustment_1') & ~(row[columns].isna())]
        row2columns[i] = set([c.split('_')[0] for c in answered])
    col2q = dict(zip(column_numbers, range(1, len(column_numbers)+1)))
    return row2columns, col2q

def process_questions(row: pd.Series, row2columns: dict, col2q: dict) -> List[dict]:
    """Process the questions for a given row."""
    
    # get the questions for this row
    idx = row.name
    questions_list = row2columns[idx]

    # get the participant-level data
    participant_data = {
        'participant_id':row['ResponseId'],
        'advisor': row['advisor'],
        'source': row['source'],
        'give_justification': row['justification'],
    }

    # preserve cols
    for col in PARTICIPANT_COLS:
        participant_data[col] = row[col]

    dataset = []

    # process each question
    for i, columnnum in enumerate(questions_list):
        columnnum = int(columnnum)
        question_data = {}
        init_total = sum([float(row[f'{columnnum}_question_{choice_num}']) for choice_num in [1, 2, 3, 4]])
        adjusted_total = sum([float(row[f'{columnnum}_adjustment_{choice_num}']) for choice_num in [1, 2, 3, 4]])
        init_time = float(row[f"{columnnum}_question_timer_Page Submit"])
        adjusted_time = float(row[f"{columnnum}_adjustment_timer_Page Submit"])
        for choice_num, letter in zip([1, 2, 3, 4], ['A', 'B', 'C', 'D']):
            question_data[f'init_choice{letter}'] = float(row[f'{columnnum}_question_{choice_num}']) / init_total
            question_data[f'adjusted_choice{letter}'] = float(row[f'{columnnum}_adjustment_{choice_num}']) / adjusted_total
        question_data['init_time'] = init_time
        question_data['adjusted_time'] = adjusted_time
        question_num = col2q[columnnum]
        question_data['question_num'] = i+1
        qualtrics = phase2qualtrics[row['phase']]
        question_meta = qualtrics[question_num]
        question_data.update(question_meta)
        question_data.update(participant_data)
        dataset.append(question_data)
    
    return dataset

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_folder", type=str, default=OUTPUT_FOLDER)
    parser.add_argument("--responses_folder", type=str, default=RESPONSES_FOLDER)
    parser.add_argument("--responses_file", type=str, default=RESPONSES_FILE)
    parser.add_argument("--min_time", type=int, default=MIN_TIME)
    parser.add_argument("--min_score", type=float, default=MIN_SCORE)
    parser.add_argument("--target_phases", type=str, nargs='+', default=TARGET_PHASES)
    
    args = parser.parse_args()
    
    survey_df = pd.read_csv(join(args.responses_folder, args.responses_file))
    survey_df, question_df = preprocess(
        survey_df,
        args.min_time,
        args.min_score,
        args.target_phases,
        args.responses_folder,
    )
    output_file = join(args.output_folder, args.responses_file.replace('.csv', '_cleaned.csv'))
    question_df.to_csv(output_file, index=False)

if __name__=="__main__":
    main()