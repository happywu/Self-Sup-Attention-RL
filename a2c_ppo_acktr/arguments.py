import argparse
from a2c_ppo_acktr.config import update_config
from a2c_ppo_acktr.config import config

import torch


def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument(
        '--cfg',
        required=True,
        type=str,
        help='experiment configure file name')
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument(
        '--eval-only', action='store_true', help='eval only')
    parser.add_argument(
        '--render', action='store_true', help='render')
    parser.add_argument(
        '--TEST_MODEL_FILE', type=str, help='model file')

    args, rest = parser.parse_known_args()
    update_config(args.cfg)

    args = parser.parse_args()
    if args.seed:
        config.seed = args.seed
    if args.TEST_MODEL_FILE:
        config.TEST.MODEL_FILE = args.TEST_MODEL_FILE

    assert config.algo in ['a2c', 'ppo', 'acktr']
    if config.recurrent_policy:
        assert config.algo in ['a2c', 'ppo'], \
            'Recurrent policy is not implemented for ACKTR'

    return args
