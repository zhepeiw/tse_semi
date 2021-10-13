"""
    data utils for the wham tse dataset
"""

import os, glob
import torch
import torchaudio
import speechbrain as sb
from speechbrain.dataio.dataset import DynamicItemDataset
from speechbrain.dataio.batch import PaddedBatch
from speechbrain.processing.signal_processing import rescale
import pandas as pd
import soundfile as sf
import numpy as np
from scipy.signal import resample_poly
import random
import pdb


def static_data_prep(hparams, part):
    assert part in ['train', 'valid', 'test']
    ds = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams['{}_data'.format(part)]
    )
    target_sr = hparams['sample_rate']
    #  dur = hparams['training_signal_len'] // target_sr
    # setting pipelines
    @sb.utils.data_pipeline.takes('mix_path', 's1_path', 's2_path', 'noise_path')
    @sb.utils.data_pipeline.provides('mix_sig', 's1_sig', 's2_sig', 'noise_sig')
    def audio_pipeline(mix_path, s1_path, s2_path, noise_path):
        mix_sig = read_and_resample(mix_path, target_sr)
        s1_sig = read_and_resample(s1_path, target_sr)
        s2_sig = read_and_resample(s2_path, target_sr)
        noise_sig = read_and_resample(noise_path, target_sr)
        # scale again to handle all signals
        sources = torch.stack([s1_sig, s2_sig, noise_sig], dim=0)
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

    @sb.utils.data_pipeline.takes('enr_path')
    @sb.utils.data_pipeline.provides('enr_sig')
    def enr_pipeline(file_path):
        sig = read_and_resample(file_path, target_sr)
        return sig

    sb.dataio.dataset.add_dynamic_item([ds], audio_pipeline)
    sb.dataio.dataset.add_dynamic_item([ds], enr_pipeline)
    # adding keys
    sb.dataio.dataset.set_output_keys(
            [ds], ["id", "mix_sig", "s1_sig", "s2_sig", "noise_sig", "enr_sig"]
        )
    return ds


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
    noise_files = get_wham_noise_filenames(hparams)
    info_dict.update(build_info_dict(noise_files))
    # setting pipelines
    @sb.utils.data_pipeline.takes('mix_path')
    @sb.utils.data_pipeline.provides('mix_sig', 's1_sig', 's2_sig', 'noise_sig',
                                     'enr_sig', 's1_clean', 's2_clean')
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
        noise_path = np.random.choice(noise_files, 1, replace=False)[0]
        noise_sig = read_wav_tensor(
            noise_path,
            int(info_dict[noise_path]['length']),
            int(info_dict[noise_path]['sr']),
            dur,
            target_sr
        )
        # scale to adjust SIR
        s1_gain = np.clip(random.normalvariate(-27.43, 2.57), -45, 0)
        s1_sig = rescale(s1_sig, torch.tensor(len(s1_sig)), s1_gain, scale='dB')
        s2_gain = np.clip(
                    s1_gain + random.normalvariate(-2.51, 2.66), -45, 0
                )
        s2_sig = rescale(s2_sig, torch.tensor(len(s2_sig)), s2_gain, scale='dB')
        sources = torch.stack([s1_sig, s2_sig], dim=0)
        mean_source_lvl = sources.abs().mean()
        mean_noise_lvl = noise_sig.abs().mean()
        noise_sig = (mean_source_lvl / mean_noise_lvl) * noise_sig
        mix_sig = torch.sum(sources, 0) + noise_sig
        # scale again to handle all signals
        max_amp = max(
            torch.abs(mix_sig).max().item(),
            *[x.item() for x in torch.abs(sources).max(dim=-1)[0]],
        )
        mix_scaling = 1 / max_amp * 0.9
        sources = mix_scaling * sources
        mix_sig = mix_scaling * mix_sig
        noise_sig = mix_scaling * noise_sig
        yield mix_sig
        for i in range(sources.shape[0]):
            yield sources[i]
        yield noise_sig
        yield enr_sig
        # clean flags
        yield torch.ones((1,)) if sp1_type == 'clean' else torch.zeros((1,))
        yield torch.ones((1,)) if sp2_type == 'clean' else torch.zeros((1,))

    sb.dataio.dataset.add_dynamic_item([ds], audio_pipeline)
    # adding keys
    sb.dataio.dataset.set_output_keys(
            [ds], ["id", "mix_sig", "s1_sig", "s2_sig", "noise_sig", "enr_sig", 's1_clean', 's2_clean']
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


########################### csv utils #########################################
def create_wham_tse_csv(wham_dir, wsj_dir, txtpath, savepath):
    '''
        args:
            wham_dir: path to wham containing 'noise', 's1', 's2', 'mix_both'
            wsj_dir: path to the wsj directory containing 'si_et_05'
            txtpath
            savepath

    '''
    all_data = []
    with open(txtpath, 'r') as f:
        idx = 0
        line = f.readline()
        while line:
            info_list = line.strip().split(' ')
            # read each signal and scale
            s1_file = info_list[0].split('/')[-1][:-len('.wv1')]
            s1_gain = float(info_list[1])
            s2_file = info_list[2].split('/')[-1][:-len('.wv1')]
            s2_gain = float(info_list[3])
            basename = '{}_{}_{}_{}.wav'.format(
                s1_file, s1_gain, s2_file, s2_gain
            )
            s1_source_path = os.path.join(wham_dir, 's1', basename)
            s2_source_path = os.path.join(wham_dir, 's2', basename)
            noise_source_path = os.path.join(wham_dir, 'noise', basename)
            mix_path = os.path.join(wham_dir, 'mix_both', basename)
            # read enrollment
            enr_file = info_list[4].replace('.wv1', '.wav')[len('wsj0/'):]
            enr_source_path = os.path.join(wsj_dir, enr_file)
            # csv entry
            all_data.append({
                'ID': idx,
                's1_path': s1_source_path,
                's2_path': s2_source_path,
                'noise_path': noise_source_path,
                'mix_path': mix_path,
                'enr_path': enr_source_path,
            })
            # next
            line = f.readline()
            idx += 1
    df = pd.DataFrame(all_data)
    df.to_csv(savepath, index=False)


########################### helper functions ##################################
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


def get_wham_noise_filenames(hparams):
    "This function lists the WHAM! noise files to be used in dynamic mixing"

    if hparams["sample_rate"] == 8000:
        noise_path = "wav8k/min/tr/noise/"
    elif hparams["sample_rate"] == 16000:
        noise_path = "wav16k/min/tr/noise/"
    else:
        raise ValueError("Unsupported Sampling Rate")

    noise_files = glob.glob(
        os.path.join(hparams["data_folder"], noise_path, "*.wav")
    )
    return noise_files
