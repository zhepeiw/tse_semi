"""
    dataloader for static mixing
"""

import torch
import speechbrain as sb
from speechbrain.dataio.dataset import DynamicItemDataset
from speechbrain.dataio.batch import PaddedBatch
from speechbrain.processing.signal_processing import rescale
import torchaudio
import soundfile as sf
import numpy as np
from scipy.signal import resample_poly
import random
import glob
import os
import pdb


def static_mixing_prep(hparams, part):
    assert part in ['train', 'valid', 'test']
    ds = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams['{}_data'.format(part)]
    )
    target_sr = hparams['sample_rate']
    dur = hparams['training_signal_len'] // target_sr
    # setting pipelines
    @sb.utils.data_pipeline.takes('s1_path', 's2_path', 'sr', 'length')
    @sb.utils.data_pipeline.provides('mix_sig', 's1_sig', 's2_sig')
    def audio_pipeline(s1_path, s2_path, sr, length):
        sr = int(sr)
        length = int(length)
        #  s1_sig = read_wav_tensor(s1_path, length, sr, dur, target_sr)
        #  s2_sig = read_wav_tensor(s2_path, length, sr, dur, target_sr)
        s1_sig = read_and_resample(s1_path, target_sr)
        s2_sig = read_and_resample(s2_path, target_sr)
        mix_sig = s1_sig + s2_sig
        # scale again to handle all signals
        sources = torch.stack([s1_sig, s2_sig], dim=0)
        #  max_amp = max(
        #      torch.abs(mix_sig).max().item(),
        #      *[x.item() for x in torch.abs(sources).max(dim=-1)[0]],
        #  )
        #  mix_scaling = 1 / max_amp * 0.9
        #  sources = mix_scaling * sources
        #  mix_sig = mix_scaling * mix_sig
        yield mix_sig
        for i in range(sources.shape[0]):
            yield sources[i]

    @sb.utils.data_pipeline.takes('enr_path', 'sr', 'enr_length')
    @sb.utils.data_pipeline.provides('enr_sig')
    def enr_pipeline(file_path, sr, length):
        sr = int(sr)
        length = int(length)
        #  sig = read_wav_tensor(file_path, length, sr, dur, target_sr)
        sig = read_and_resample(file_path, target_sr)
        return sig

    sb.dataio.dataset.add_dynamic_item([ds], audio_pipeline)
    sb.dataio.dataset.add_dynamic_item([ds], enr_pipeline)
    # adding keys
    sb.dataio.dataset.set_output_keys(
            [ds], ["id", "mix_sig", "s1_sig", "s2_sig", "enr_sig"]
        )
    return ds
    #  loader = torch.utils.data.DataLoader(
    #      ds,
    #      batch_size=hparams["dataloader_opts"]["batch_size"],
    #      num_workers=hparams["dataloader_opts"]["num_workers"],
    #      shuffle=True if part == 'train' else False,
    #      collate_fn=PaddedBatch,
    #      worker_init_fn=lambda x: np.random.seed(
    #          int.from_bytes(os.urandom(4), "little") + x
    #      ),
    #  )
    #  return loader


def dynamic_mixing_prep(hparams, part):
    target_sr = hparams['sample_rate']
    dur = hparams['training_signal_len'] // target_sr
    assert part in ['train', 'valid', 'test']
    ds = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams['{}_data'.format(part)]
    )
    # process clean and unclean corpus
    sp_dict = {'clean': {}, 'unclean': {}}
    info_dict = {}
    sp_list = {'clean': [], 'unclean': []}
    sp_weights = {'clean': [], 'unclean': []}
    for folder_info in hparams['base_folder_dm_info_list']:
        file_list = glob.glob(os.path.join(folder_info['path'], '**/*.{}'.format(folder_info['ext'])), recursive=True)
        curr_sp_dict, _ = build_sp_dict(file_list)
        sp_dict[folder_info['type']].update(curr_sp_dict)
        info_dict.update(build_info_dict(file_list))
    for clean_type in sp_dict.keys():
        sp_list[clean_type] = [x for x in sp_dict[clean_type].keys()]
        curr_sp_weights = [len(sp_dict[clean_type][spid]) for spid in sp_dict[clean_type].keys()]
        curr_sp_weights = [w / sum(curr_sp_weights) for w in curr_sp_weights]
        sp_weights[clean_type] = curr_sp_weights

    #  if hparams['overfit_utt'] is not None:
    #      print('Overfitting dataset with size {}'.format(hparams['overfit_utt']))
    #      file_list = random.sample(file_list, hparams['overfit_utt'])
    #  sp_dict, sp_weights = build_sp_dict(file_list)
    #  sp_list = [x for x in sp_dict.keys()]
    #  sp_weights = [w / sum(sp_weights) for w in sp_weights]
    #  info_dict = build_info_dict(file_list)
    #  pdb.set_trace()
    # setting pipelines
    @sb.utils.data_pipeline.takes('mix_path')
    @sb.utils.data_pipeline.provides('mix_sig', 's1_sig', 's2_sig', 'enr_sig',
                                     's1_clean', 's2_clean')
    def audio_pipeline(file_path):
        # first, decide type of each speaker
        sp1_type = 'clean' if np.random.rand() < hparams['data_clean_prob'] else 'unclean'
        sp2_type = 'clean' if np.random.rand() < hparams['data_clean_prob'] else 'unclean'
        for attempts in range(10):
            # in case of IO failure
            try:
                if sp1_type == sp2_type:
                    sp1, sp2 = np.random.choice(sp_list[sp1_type], 2, replace=False, p=sp_weights[sp1_type])
                else:
                    sp1 = np.random.choice(sp_list[sp1_type], 1, replace=False, p=sp_weights[sp1_type])[0]
                    sp2 = np.random.choice(sp_list[sp2_type], 1, replace=False, p=sp_weights[sp2_type])[0]
                enr_path, s1_path = np.random.choice(sp_dict[sp1_type][sp1], 2, replace=True)
                s2_path = np.random.choice(sp_dict[sp2_type][sp2], 1, replace=False)[0]
                s1_sig = read_wav_tensor(s1_path,
                                        int(info_dict[s1_path]['length']),
                                        int(info_dict[s1_path]['sr']),
                                        dur,
                                        target_sr)
                s2_sig = read_wav_tensor(s2_path,
                                        int(info_dict[s2_path]['length']),
                                        int(info_dict[s2_path]['sr']),
                                        dur,
                                        target_sr)
                enr_sig = read_wav_tensor(enr_path,
                                         int(info_dict[enr_path]['length']),
                                         int(info_dict[enr_path]['sr']),
                                         dur,
                                         target_sr)
                break
            except:
                if attempts == 9:
                    pdb.set_trace()
                continue
        # scale to adjust SIR
        s1_gain = np.clip(random.normalvariate(-27.43, 2.57), -45, 0)
        s1_sig = rescale(s1_sig, torch.tensor(len(s1_sig)), s1_gain, scale='dB')
        s2_gain = np.clip(
                    s1_gain + random.normalvariate(-2.51, 2.66), -45, 0
                )
        s2_sig = rescale(s2_sig, torch.tensor(len(s2_sig)), s2_gain, scale='dB')
        mix_sig = s1_sig + s2_sig
        # scale again to handle all signals
        sources = torch.stack([s1_sig, s2_sig], dim=0)
        max_amp = max(
            torch.abs(mix_sig).max().item(),
            *[x.item() for x in torch.abs(sources).max(dim=-1)[0]],
        )
        mix_scaling = 1 / max_amp * 0.9
        sources = mix_scaling * sources
        mix_sig = mix_scaling * mix_sig
        yield mix_sig
        for i in range(sources.shape[0]):
            yield sources[i]
        yield enr_sig
        # clean flags
        yield torch.ones((1,)) if sp1_type == 'clean' else torch.zeros((1,))
        yield torch.ones((1,)) if sp2_type == 'clean' else torch.zeros((1,))

    sb.dataio.dataset.add_dynamic_item([ds], audio_pipeline)
    # adding keys
    sb.dataio.dataset.set_output_keys(
            [ds], ["id", "mix_sig", "s1_sig", "s2_sig", "enr_sig", 's1_clean', 's2_clean']
        )
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=hparams["{}_dataloader_opts".format(part)]["batch_size"],
        num_workers=hparams["{}_dataloader_opts".format(part)]["num_workers"],
        collate_fn=PaddedBatch,
        worker_init_fn=lambda x: np.random.seed(
            int.from_bytes(os.urandom(4), "little") + x
        ),
    )
    return loader


'''
    helper functions
'''
def read_wav_tensor(file_path, length, sr, dur, target_sr):
    '''
        args:
            file_path: path to audio file
            length: length of the audio file (in samples)
            sr: sample rate of audio file
            dur: target duration (in sec)
            target sr: target sample rate

        out:
            1d torch Tensor
    '''
    seg_len = int(sr * dur)
    frames = seg_len if seg_len < length else -1
    start = np.random.randint(0, length - seg_len + 1) if frames != -1 else 0
    sig, _ = sf.read(file_path, frames=frames, start=start, dtype='float32', always_2d=True)
    sig = sig[:, 0]
    if len(sig) < seg_len:
        tmp = np.zeros(seg_len).astype(np.float32)
        rstart = np.random.randint(0, seg_len - len(sig) + 1)
        tmp[rstart:rstart+len(sig)] = sig
        sig = tmp
    if sr != target_sr:
        sig = resample_poly(sig, target_sr, sr).astype(np.float32)
    return torch.from_numpy(sig)


def read_and_resample(path, target_sr):
    sig, sr = sf.read(path, dtype='float32', always_2d=True)
    sig = sig[:, 0]
    if sr != target_sr:
        sig = resample_poly(sig, target_sr, sr).astype(np.float32)
    return torch.from_numpy(sig)


def build_sp_dict(file_list):
    sp_dict = {}
    for path in file_list:
        #  spid = path.split('/')[-3]
        if 'wsj' in path:
            spid = path.split('/')[-2]
        else:
            spid = path.split('/')[-3]
        if spid not in sp_dict:
            sp_dict[spid] = [path]
        else:
            sp_dict[spid].append(path)
    sp_weights = [len(sp_dict[spid]) for spid in sp_dict.keys()]
    return sp_dict, sp_weights


def build_info_dict(file_list):
    info_dict = {}
    for path in file_list:
        info_dict[path] = {}
        info = torchaudio.info(path)
        info_dict[path]['sr'] = info.sample_rate
        info_dict[path]['length'] = info.num_frames
    return info_dict
