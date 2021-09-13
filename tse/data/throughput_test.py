from data_mixing import dynamic_mixing_prep
import torch
import numpy as np
from speechbrain.dataio.batch import PaddedBatch

hyp = {
    'dataloader_opts': {
        'batch_size': 1,
        'num_workers': 3,
    },
    'overfit_utt': None,
    'train_data': '/mnt/data/zhepei/outputs/sb_tse/results/2021-09-05+22-13-31+seed_1234+sepformer_wsj_16k/save/dummy.csv',
    'base_folder_dm': '/mnt/nvme/wsj0.8k/si_tr_s/',
    'base_folder_dm_ext': 'wav',
    'sample_rate': 8000,
    'training_signal_len': 24000,
}

dl = dynamic_mixing_prep(hyp, 'train')
device = torch.device('cuda:0')
import time
tstart = time.time()
for i, batch in enumerate(dl):
    mix_sig = batch['mix_sig'][0]
    mix_sig = mix_sig.to(device)
    if (i+1) % 20 == 0:
        print('avg per batch: {}'.format((time.time() - tstart) / (i+1)))
