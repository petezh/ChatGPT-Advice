#!/bin/bash

pip install -r requirements.txt
python scripts/create_dataset.py --num_questions 50 --splits test dev val
python scripts/evaluate.py --mode cot