import os
import time
import logging
import argparse
import datetime

from experiments import experiments_spid
EXP_MODULES = [experiments_spid]

from utils.writer import MyWriter, WandbWriter
from utils.train import train

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
    args = parser.parse_args()
    experiment_name, run_name, hyp = get_exp_data(args.exp)
    if args.debug:
        print('\nApplying debug hyp updates: \n {}'.format(DEBUG_HYP_UPDATES))
        hyp.update(DEBUG_HYP_UPDATES)
    gpus = [0]
    if len(gpus) >= 4:
        print('Adjusting batch size to multi-gpu training...')
        hyp['train/batch_size'] *= 4
        hyp['train/num_workers'] *= 4
    elif len(gpus) >= 2:
        print('Adjusting batch size to multi-gpu training...')
        hyp['train/batch_size'] *= 2
        hyp['train/num_workers'] *= 2
    run_stamp = '{}_{}'.format(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'), run_name)
    if args.debug:
        run_stamp = 'debug_' + run_stamp
    pt_dir = os.path.join(hyp['log/base_dir'], experiment_name, run_stamp, hyp['log/chkpt_dir'])
    os.makedirs(pt_dir, exist_ok=True)

    log_dir = os.path.join(hyp['log/base_dir'], experiment_name, run_stamp)
    os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir,
                '%s.log' % (run_stamp))),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger()

    if hyp['log/use_wandb']:
        writer = WandbWriter(hyp, run_stamp)
    else:
        writer = MyWriter(hyp, log_dir)

    from datasets.pt_datasets import create_dataloader
    _, trainloader = create_dataloader(hyp)

    chkpt_path, hp_str = None, None
    testloader = None
    train(args, pt_dir, chkpt_path, trainloader, testloader, writer, logger, hyp, hp_str)
