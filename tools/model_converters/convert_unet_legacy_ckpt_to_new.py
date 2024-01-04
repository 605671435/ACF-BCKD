# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
from pathlib import Path

import torch
from seg.utils.misc import find_best_in_multi_run

def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert KD checkpoint to student-only checkpoint')
    parser.add_argument('work_dir', help='input checkpoint filename')
    args = parser.parse_args()
    return args


def main(ckpt):
    checkpoint = torch.load(ckpt, map_location='cpu')
    new_state_dict = dict()
    new_meta = checkpoint['meta']

    for key, value in checkpoint['state_dict'].items():
        new_key = key.replace('dsa_block', 'DSA')
        new_state_dict[new_key] = value

    checkpoint = dict()
    checkpoint['meta'] = new_meta
    checkpoint['state_dict'] = new_state_dict

    torch.save(checkpoint, ckpt)


if __name__ == '__main__':
    args = parse_args()
    run_dir = [dir for dir in os.listdir(args.work_dir) if dir.startswith('run')]
    assert len(run_dir) == 3
    for run in run_dir:
        ckpt = [file for file in os.listdir(os.path.join(args.work_dir, run)) if file.startswith('best')][0]
        main(os.path.join(args.work_dir, run, ckpt))
