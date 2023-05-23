#!/usr/bin/env python3

from argparse import ArgumentParser
from audio_play import say
from audio_record import record_to_file
import logging
import recognizer
from transcribe import transcribe_from_audio_file


logger = logging.getLogger(__name__)


def output(text, prefix=''):
    logger.info(prefix+text)
    # say(text, 'ai-output.mp3')


parser = ArgumentParser()
parser.add_argument('-v','--verbose', action='store_true')
parser.add_argument('-d','--domain', help='The command domain to use, e.g. volvo')
options = parser.parse_args()
if options.domain:
    recognizer.cmd_filename = ('cmds/%s.json'%options.domain)

if options.verbose:
    logging.basicConfig(format='%(asctime)-15s %(name)-24s %(levelname)-7s %(message)s', level=logging.DEBUG)
else:
    logging.basicConfig(format='%(message)s', level=logging.INFO)

logger.debug('loading commands')
cmd_tree = recognizer.load_cmd_tree()
unembedded_cmds = recognizer.filter_unembedded_cmds(cmd_tree)
if unembedded_cmds:
    logger.debug('updating commands %s', unembedded_cmds)
    recognizer.update_embeddings(cmd_tree, unembedded_cmds)
    recognizer.save_cmd_tree(cmd_tree)


output('How may I be of service?')
i = 0
while True:
    # fn = 'human-input.wav'
    # record_to_file(fn)
    # cmd = transcribe_from_audio_file(fn)
    cmd = input().strip()
    if not cmd:
        output(prefix="I couldn't quite catch that. ", text='Please repeat.')
        continue
    logger.debug('input command: %s', cmd)
    cmd_embedding = recognizer.embed(cmd)
    cmd_key, best_cmd, diff = recognizer.find_closest_command(cmd_tree, cmd, cmd_embedding)
    logger.debug('cmd=%s, diff=%f, key=%s', best_cmd, diff, cmd_key)
    if best_cmd and diff > 0.91:
        output(prefix='running command: ', text=best_cmd)
    else:
        output(prefix='unknown command: ', text=cmd)
