import json
import numpy as np
import os
import openai
import logging
from random import choice


cmd_filename = 'cmds/generic.json'
logger = logging.getLogger(__name__)
openai.api_key = os.getenv('OPENAI_API_KEY')


if not openai.api_key:
    logger.error('you need to a) create an OpenAI account, b) create API keys, and c) set the OPENAI_API_KEY system variable')
    os.exit(1)


def load_cmd_tree(filename=None):
    cmd_tree = json.load(open(filename or cmd_filename))
    return cmd_tree


def save_cmd_tree(cmd_tree, filename=None):
    j = json.dumps(cmd_tree, indent='  ')
    j = j.replace('\n      ', '').replace('\n    ]', ']')
    open(filename or cmd_filename, 'wt').write(j)


def filter_unembedded_cmds(cmd_tree):
    unembedded = [k for kv in cmd_tree.values() for k,v in kv.items() if not v]
    return unembedded


def update_embeddings(cmd_tree, cmds):
    for cmd,embedding in zip(cmds, _get_embeddings(cmds)):
        for kv in cmd_tree.values():
            if cmd in kv:
                kv[cmd] = embedding


def embed(cmd):
    embeddings = _get_embeddings([cmd])
    return embeddings[0]


def find_closest_command(cmd_tree, orig_cmd, cmd_embedding):
    flat = [{'cmd':c, 'diff':cosine_similarity(e, cmd_embedding), 'key':k} for k,cmds in cmd_tree.items() for c,e in cmds.items()]
    closest = sorted(flat, key=lambda o: -o['diff'])
    logger.debug('closest commands to %s: %s', orig_cmd, closest[:3])
    return closest[0]['key'], closest[0]['cmd'], closest[0]['diff']


def _get_embeddings(list_of_text, engine='text-similarity-babbage-001'):
    assert len(list_of_text) <= 2048, 'The batch size should not be larger than 2048.'
    list_of_text = [text.replace('\n', ' ') for text in list_of_text]
    data = openai.Embedding.create(input=list_of_text, engine=engine).data
    data = sorted(data, key=lambda x: x['index'])
    return [d['embedding'] for d in data]


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
