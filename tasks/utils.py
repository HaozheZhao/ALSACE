from tasks.xglue.dataset import task_to_keys as xglue_tasks
from tasks.xtreme.dataset import task_to_keys as xtreme_tasks

XGLUE_DATASETS = list(xglue_tasks.keys())+['mlama','geolama']
XTREME_DATASETS = list(xtreme_tasks.keys())
NER_DATASETS = ["conll2003", "conll2004", "ontonotes"]
SRL_DATASETS = ["conll2005", "conll2012"]
QA_DATASETS = ["squad", "squad_v2",'mlama','geolama']


TASKS = ["xglue","xtreme",'mlama','geolama']

DATASETS = XGLUE_DATASETS+XTREME_DATASETS

ADD_PREFIX_SPACE = {
    'bert': False,
    'roberta': True,
    'deberta': True,
    'gpt2': True,
    'deberta-v2': True,
}

USE_FAST = {
    'bert': True,
    'roberta': True,
    'deberta': True,
    'gpt2': True,
    'deberta-v2': False,
}