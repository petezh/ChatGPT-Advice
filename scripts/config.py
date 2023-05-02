import yaml

config = yaml.load(open('config.yaml'), Loader=yaml.FullLoader)
topics = [c['topic'] for c in config]
topic2display = {c['topic']: c['display'] for c in config}
topic2category = {c['topic']: c['supercategory'] for c in config}
topic2fam_topic = {c['topic']: c['fam_topic'] for c in config}
category2display = {
    'stem':'STEM',
    'humanities':'Humanities',
    'social sciences':'Social Sciences',
}