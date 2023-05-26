import json
import numpy as np
import os
import openai
import logging
from random import choice
import re


meta_filename = 'cmds/meta-cmds.json'
cmd_filename = 'cmds/generic-cmds.json'
max_phrase_words = 3

logger = logging.getLogger(__name__)
openai.api_key = os.getenv('OPENAI_API_KEY')


if not openai.api_key:
    logger.error('you need to a) create an OpenAI account, b) create API keys, and c) set the OPENAI_API_KEY system variable')
    os._exit(1)


class rdict(dict):
    pass


def load_cmd_tree(filename=None):
    cmd_tree = json.load(open(filename or cmd_filename))
    cmd_tree = rdict(cmd_tree)
    cmd_tree.meta = json.load(open(meta_filename))
    return cmd_tree


def save_cmd_tree(cmd_tree, filename=None):
    _dump_json_embeddings(cmd_tree, filename or cmd_filename)
    _dump_json_embeddings(cmd_tree.meta, meta_filename)


def filter_unembedded_cmds(cmd_tree):
    unembedded = []
    unembedded += [k for kv in cmd_tree.values() for k,v in kv.items() if not v]
    unembedded += [k for kv in cmd_tree.meta.values() for k,v in kv.items() if not v]
    return unembedded


def update_embeddings(cmd_tree, cmds):
    for cmd,embedding in zip(cmds, _get_embeddings(cmds)):
        for kv in cmd_tree.values():
            if cmd in kv:
                kv[cmd] = embedding
        for kv in cmd_tree.meta.values():
            if cmd in kv:
                kv[cmd] = embedding


def find_closest_command(cmd_tree, cmd):
    phrases = _str_to_phrases(cmd, nwords=max_phrase_words)
    cmd = _strip_exact_meta(cmd_tree.meta['filter_words'].keys(), cmd)
    phrases = _str_to_phrases(cmd, nwords=max_phrase_words)
    key,phrase,diff = _phrase_exact_match(cmd_tree, phrases)
    if diff > 0.97 and (len(phrases) == 1 or len(phrase.split()) >= 2):
        logger.debug('exact command match %s: %s (%i)', cmd, phrase, diff*100)
        return key, phrase, diff

    logger.debug('phrases remaining: %s', phrases)
    embeddings = _get_embeddings(phrases)
    flat = [{'cmd':c, 'diff':_cosine_similarity(e, embedding), 'key':k} for embedding in embeddings for k,cmds in cmd_tree.items() for c,e in cmds.items()]
    closest = sorted(flat, key=lambda o: -o['diff']-(len(o['cmd'].split())-1)*0.1)
    logger.debug('closest commands to %s: %s', cmd, closest[:3])
    return closest[0]['key'], closest[0]['cmd'], closest[0]['diff']


def _strip_exact_meta(phrases, cmd):
    for phrase in phrases:
        # logger.debug(f'stripping "{phrase}" from {cmd}')
        cmd = re.sub(f'\\b{phrase}\\b', '', cmd)
        cmd = cmd.strip().replace('  ', ' ')
        # logger.debug(f'cmd = "{cmd}"')
    return cmd


def _phrase_exact_match(cmd_tree, phrases):
    for phrase in phrases:
        for k,cmds in cmd_tree.items():
            for c in cmds:
                if c == phrase:
                    return k, c, 1.0
    return None, None, 0.0


def _dump_json_embeddings(tree, filename):
    j = json.dumps(tree, indent='  ')
    j = j.replace('\n      ', '').replace('\n    ]', ']')
    open(filename, 'wt').write(j)


def _get_embeddings(list_of_text, engine='text-similarity-babbage-001'):
    assert len(list_of_text) <= 2048, 'The batch size should not be larger than 2048.'
    list_of_text = [text.replace('\n', ' ') for text in list_of_text]
    data = openai.Embedding.create(input=list_of_text, engine=engine).data
    data = sorted(data, key=lambda x: x['index'])
    return [d['embedding'] for d in data]


def _str_to_phrases(cmd, nwords=2):
    words = cmd.split()
    phrases = [' '.join(words[i:j]) for i in range(len(words)) for j in range(min(len(words),i+nwords), i, -1)]
    return sorted(phrases, key=lambda w:-len(w))


def _cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
