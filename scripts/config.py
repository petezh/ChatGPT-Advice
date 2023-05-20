"""
Configuration file for the project.
"""

import yaml

config = yaml.load(open('config.yaml'), Loader=yaml.FullLoader)
topics = [c['topic'] for c in config["mmlu_topics"]]
topic2display = {c['topic']: c['display'] for c in config["mmlu_topics"]}
topic2category = {c['topic']: c['supercategory'] for c in config["mmlu_topics"]}
topic2fam_topic = {c['topic']: c['fam_topic'] for c in config["mmlu_topics"]}
category2display = {
    'stem':'STEM',
    'humanities':'Humanities',
    'social sciences':'Social Sciences',
}
fam_topics = config["fam_topics"]

comfort_map = { # mapping of comfort level to numerical value
    'Uncomfortable': -1,
    'Neutral': 0,
    'Comfortable': 1,
}