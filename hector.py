#!/usr/bin/env python3

from argparse import ArgumentParser
from audio_play import say
from audio_record import record_to_file
import logging
import recognizer
from transcribe import transcribe_from_audio_file
from brow import web_executor


logger = logging.getLogger(__name__)


def output(text, *args, **kwargs):
    prefix = kwargs.get('prefix', '')
    logger.info(' '.join([prefix+text]+[str(a) for a in args]))
    if options.audio:
        say(text, 'ai-output.mp3')


parser = ArgumentParser()
parser.add_argument('-v','--verbose', action='store_true')
parser.add_argument('-d','--domain', required=True, help='The command domain to use, e.g. volvo-connect')
parser.add_argument('-a','--audio', action='store_true', help='Use speech to control and listen through speakers')
options = parser.parse_args()
if options.domain:
    recognizer.cmd_filename = ('cmds/%s-cmds.json'%options.domain)
    web_executor.cmd_filename = ('cmds/%s-actions.json'%options.domain)

if options.verbose:
    logging.basicConfig(format='%(asctime)-15s %(name)-24s %(levelname)-7s %(message)s', level=logging.DEBUG)
else:
    logging.basicConfig(format='%(message)s', level=logging.INFO)

logger.debug('loading commands')
cmd_tree = recognizer.load_cmd_tree()
unembedded_cmds = recognizer.filter_unembedded_cmds(cmd_tree)
if unembedded_cmds:
    logger.info('updating commands %s', unembedded_cmds)
    recognizer.update_embeddings(cmd_tree, unembedded_cmds)
    recognizer.save_cmd_tree(cmd_tree)


actions = web_executor.load_actions()
web_executor.run(actions, 'start')
output('How may I be of service?')
while True:
    if options.audio:
        fn = 'human-input.wav'
        record_to_file(fn)
        cmd = transcribe_from_audio_file(fn)
    else:
        cmd = input().strip()
    for ch in '.,!?':
        cmd = cmd.replace(ch,' ')
    cmd = cmd.strip().replace('  ', ' ').lower()
    if not cmd:
        output(prefix="I couldn't quite catch that. ", text='Please repeat.')
        continue
    logger.debug('input command: %s', cmd)
    cmd_key, best_cmd, diff = recognizer.find_closest_command(cmd_tree, cmd)
    logger.debug('cmd=%s, diff=%f, key=%s', best_cmd, diff, cmd_key)
    if cmd_key and best_cmd and diff > 0.92:
        web_executor.run(actions, cmd_key)
        output(prefix=f'running command ({int(diff*100)}% accuracy): ', text=best_cmd)
    else:
        output('unknown command:', cmd, 'best guess:', best_cmd, 'diff:', diff)
