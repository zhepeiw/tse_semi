import os, math
import torch
import torch.nn as nn
import numpy as np
import traceback

from modules.embedder import Embedder
from losses.ge2e import GE2ELoss
from itertools import chain

import utils.evaluate as evaluate
import pdb

def train(args, pt_dir, chkpt_path, trainloader, testloader, writer, logger, hp, hp_str):
    # device setup
    gpus = [0]
    if len(gpus) > 0:
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    if len(gpus) > 1:
        embedder = nn.DataParallel(embedder)
        loss_fn = nn.DataParallel(loss_fn)
        print('Data parallel training')
    # model
    embedder = Embedder(hp)
    embedder = embedder.to(device)
    if hp['log/use_wandb']:
        writer.watch(embedder)
    # criterion
    loss_fn = GE2ELoss('softmax')
    loss_fn = loss_fn.to(device)

    if hp['train/optimizer'] == 'adam':
        optimizer = torch.optim.Adam(chain(embedder.parameters(), loss_fn.parameters()),
                                     lr=hp['train/adam'])
    elif hp['train/optimizer'] == 'sgd':
        optimizer = torch.optim.SGD([
            {'params': embedder.parameters()},
            {'params': loss_fn.parameters()},
        ], lr=hp['train/sgd'])
    else:
        raise Exception("%s optimizer not supported" % hp['train/optimizer'])

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     hp['train/scheduler/milestones'],
                                                     hp['train/scheduler/gamma'])
    step = 0
    cache_loss = []

    if hp['pretrain']:
        ckpt = torch.load(hp['pretrain/ckpt_path'])
        embedder.load_state_dict(ckpt['embedder'])
        loss_fn.load_state_dict(ckpt['loss_fn'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        step = ckpt['step']
        print('Checkpoint loaded from {}'.format(hp['pretrain/ckpt_path']))

    try:
        while True:
            embedder.train()
            for batch in trainloader:
                # data
                batch = batch.to(device)  # [B, 1, M*L]
                model_inp = batch.view(batch.shape[0]*hp['datasets/num_utterances_per_speaker_per_batch'], 1, -1)  # [B*M, 1, L]
                optimizer.zero_grad()
                # forward
                model_emb = embedder(model_inp)  # [B*M, D]
                model_emb = model_emb.view(batch.shape[0], hp['datasets/num_utterances_per_speaker_per_batch'],
                                           model_emb.shape[-1])  # [B, M, D]
                # loss
                loss = loss_fn(model_emb)
                loss.backward()
                if hp['train/clip_grad_norm'] is not None:
                    nn.utils.clip_grad_norm_(embedder.parameters(), hp['train/clip_grad_norm'])
                    nn.utils.clip_grad_norm_(loss_fn.parameters(), 1.)
                optimizer.step()
                scheduler.step()
                step += 1

                loss = loss.item()
                cache_loss.append(loss)
                if loss > 1e8 or math.isnan(loss):
                    pdb.set_trace()
                    logger.error("Loss exploded to %.02f at step %d!" % (loss, step))
                    raise Exception("Loss exploded")

                # write loss to tensorboard
                if step % hp['train/summary_interval'] == 0:
                    writer.log_training(np.mean(cache_loss), step, None)
                    logger.info("Wrote summary at step {}: loss {:.6f}".format(step, np.mean(cache_loss)))
                    cache_loss = []

                if step % hp['eval/eval_interval'] == 0:
                    eer = evaluate.evaluate_verification(embedder, hp,
                                                     input_size=int(hp['audio/sample_rate']*hp['datasets/duration']))
                    writer.log_evaluation(eer, step)

                if step % hp['train/checkpoint_interval'] == 0:
                    save_path = os.path.join(pt_dir, 'chkpt_%d.pt' % step)
                    torch.save({
                        'embedder': embedder.state_dict(),
                        'loss_fn': loss_fn.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'step': step,
                        'eer': eer,
                        'hp': hp,
                    }, save_path)


    except Exception as e:
        logger.info("Exiting due to exception: %s" % e)
        traceback.print_exc()
