import numpy as np
from tensorboardX import SummaryWriter
import wandb
import pdb


class MyWriter(SummaryWriter):
    def __init__(self, hp, logdir):
        super(MyWriter, self).__init__(logdir)
        self.hp = hp

    def log_training(self, train_loss, step, loss_val_dict=None):
        self.add_scalar('train_loss', train_loss, step)
        if loss_val_dict is not None:
            for loss_name, loss_val in loss_val_dict.items():
                self.add_scalar('train_loss_{}'.format(loss_name), loss_val.item(), step)

    def log_evaluation(self, eer, step):
        self.add_scalar('EER', eer, step)


class WandbWriter():
    def __init__(self, hyp, run_stamp):
        wandb.init(
            project=hyp['log/wandb/project'],
            entity=hyp['log/wandb/entity'],
            dir=hyp['log/base_dir'],
            name=run_stamp,
            config=hyp
        )
        self.hyp = hyp

    def watch(self, model):
        wandb.watch(model, log_freq=self.hyp['train/summary_interval'])

    def log_training(self, train_loss, step, loss_val_dict=None):
        wandb.log({'train_loss': train_loss}, step=step)
        if loss_val_dict is not None:
            for loss_name, loss_val in loss_val_dict.items():
                wandb.log({
                    'train_loss_{}'.format(loss_name): loss_val.item(),
                }, step=step)

    def log_evaluation(self, eer, step):
        wandb.log({'EER': eer}, step=step)
