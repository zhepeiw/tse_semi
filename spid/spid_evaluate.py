import os
import time
import logging
import argparse
import datetime
import torch

from experiments import experiments_spid
EXP_MODULES = [experiments_spid]

import utils.evaluate as evaluate
from modules.embedder import Embedder

DEBUG_HYP_UPDATES = {
    'train/num_workers': 0,
    'train/pin_memory': False,
    'train/summary_interval': 10,
    'seed': None,
    'eval/eval_interval': 2,
    'eval/num_files': 10,
    'log/use_wandb': False,
}


def get_exp_data(exp):
    """ calls the function with the same name as exp and returns
        the experiment's settings
    """
    exp_data_funcs = []
    if exp in globals():
        exp_data_funcs.append(globals()[exp])

    for modu in EXP_MODULES:
        if exp in dir(modu):
            exp_data_funcs.append(getattr(modu, exp))

    assert len(exp_data_funcs) == 1, "There needs to be exactly one function with the exp name"
    return exp_data_funcs[0]() # will return experiment_name, run_name, hyp


if __name__ == '__main__':
    import pdb
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--exp', type=str, required=True,
                        help="exp function for configuration")
    parser.add_argument("--debug", dest='debug', action='store_true')
    parser.add_argument('--pretrained_path', type=str)
    args = parser.parse_args()
    experiment_name, run_name, hyp = get_exp_data(args.exp)
    if args.debug:
        print('\nApplying debug hyp updates: \n {}'.format(DEBUG_HYP_UPDATES))
        hyp.update(DEBUG_HYP_UPDATES)
    gpus = [0]
    if len(gpus) > 0:
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    # model
    embedder = Embedder(hyp)
    if args.pretrained_path is not None:
        embedder.load_state_dict(torch.load(args.pretrained_path, map_location='cpu')['embedder'])
    embedder = embedder.to(device)
    eer = evaluate.evaluate_verification(embedder, hyp,
                                     input_size=int(hyp['audio/sample_rate']*hyp['datasets/duration']))
