"""
    csv prepratation for static mixture creation
"""

import torch
import torchaudio
import pandas as pd
import speechbrain
import os, glob
import soundfile as sf
import numpy as np
import random
from scipy.signal import resample_poly
from speechbrain.processing.signal_processing import rescale
import pdb

"""
    helper functions
"""

def read_wav_tensor(file_path, length, sr, dur, target_sr):
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


def build_sp_dict(file_list):
    sp_dict = {}
    for path in file_list:
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

def generate_static_data_files(data_dir, ext, save_dir, limit=100, dur=3., target_sr=16000):
    os.makedirs(save_dir, exist_ok=True)
    file_list = glob.glob(os.path.join(data_dir, '**/*.{}'.format(ext)),
                         recursive=True)
    sp_dict, sp_weights = build_sp_dict(file_list)
    sp_list = [x for x in sp_dict.keys()]
    sp_weights = [w / sum(sp_weights) for w in sp_weights]
    info_dict = build_info_dict(file_list)
    all_data = []
    idx = 0
    while idx < limit:
        sp1, sp2 = np.random.choice(sp_list, 2, replace=False, p=sp_weights)
        enr_path, s1_path = np.random.choice(sp_dict[sp1], 2, replace=True)
        s2_path = np.random.choice(sp_dict[sp2], 1, replace=False)[0]
        # read and scale the signals
        s1_sig = read_wav_tensor(s1_path,
                                int(info_dict[s1_path]['length']),
                                int(info_dict[s1_path]['sr']),
                                dur,
                                target_sr)
        s1_gain = np.clip(random.normalvariate(-27.43, 2.57), -45, 0)
        s1_sig = rescale(s1_sig, torch.tensor(len(s1_sig)), s1_gain, scale='dB')
        s2_sig = read_wav_tensor(s2_path,
                                int(info_dict[s2_path]['length']),
                                int(info_dict[s2_path]['sr']),
                                dur,
                                target_sr)
        s2_gain = np.clip(
                    s1_gain + random.normalvariate(-2.51, 2.66), -45, 0
                )
        s2_sig = rescale(s2_sig, torch.tensor(len(s2_sig)), s2_gain, scale='dB')
        mix_sig = s1_sig + s2_sig
        sources = torch.stack([s1_sig, s2_sig], dim=0)
        # rescale to avoid clipping
        max_amp = max(
            torch.abs(mix_sig).max().item(),
            *[x.item() for x in torch.abs(sources).max(dim=-1)[0]],
        )
        mix_scaling = 1 / max_amp * 0.9
        sources = mix_scaling * sources
        mix_sig = mix_scaling * mix_sig
        s1_sig = sources[0]
        s2_sig = sources[1]
        # save to disk
        sp1_flag = '-'.join(s1_path.split('/')[-3:])[:-len('.{}'.format(ext))]
        sp2_flag = '-'.join(s2_path.split('/')[-3:])[:-len('.{}'.format(ext))]
        print('{}: Generating files for {}, {}'.format(idx, sp1_flag, sp2_flag))
        s1_out_path = os.path.join(save_dir, 'idx{:04}_{}_{}_s1.wav'.format(idx, sp1_flag, sp2_flag))
        sf.write(s1_out_path, s1_sig.numpy(), target_sr)
        s2_out_path = os.path.join(save_dir, 'idx{:04}_{}_{}_s2.wav'.format(idx, sp1_flag, sp2_flag))
        sf.write(s2_out_path, s2_sig.numpy(), target_sr)
        mix_out_path = os.path.join(save_dir, 'idx{:04}_{}_{}_mix.wav'.format(idx, sp1_flag, sp2_flag))
        sf.write(mix_out_path, mix_sig.numpy(), target_sr)
        # dont rescale enrollment
        enr_out_path = os.path.join(save_dir, 'idx{:04}_{}_{}_enr.wav'.format(idx, sp1_flag, sp2_flag))
        enr_sig = read_wav_tensor(enr_path,
                                 int(info_dict[enr_path]['length']),
                                 int(info_dict[enr_path]['sr']),
                                 dur,
                                 target_sr)
        sf.write(enr_out_path, enr_sig.numpy(), target_sr)
        # csv entry
        all_data.append({
            'ID': idx,
            'descriptor': 'idx{:04}_{}_{}'.format(idx, sp1_flag, sp2_flag),
            'mix_path': mix_out_path,
            's1_path': s1_out_path,
            's2_path': s2_out_path,
            'enr_path': enr_out_path,
            'sr': target_sr,
            'length': len(mix_sig),
            'enr_length': len(enr_sig),
        })
        idx += 1
    df = pd.DataFrame(all_data)
    df.to_csv(os.path.join(save_dir, 'meta.csv'), index=False)


def wsj_tse_remix(datapath, txtpath, save_dir, limit=None, target_sr=8000):
    os.makedirs(os.path.join(save_dir, 's1'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 's2'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'enr'), exist_ok=True)

    def read_and_resample(path):
        sig, sr = sf.read(path, dtype='float32', always_2d=True)
        sig = sig[:, 0]
        if sr != target_sr:
            sig = resample_poly(sig, target_sr, sr).astype(np.float32)
        return torch.from_numpy(sig)

    all_data = []
    with open(txtpath, 'r') as f:
        idx = 0
        line = f.readline()
        while line:
            info_list = line.strip().split(' ')
            # read each signal and scale
            s1_file = info_list[0].replace('.wv1', '.wav')[len('wsj0/'):]
            s1_source_path = os.path.join(datapath, s1_file)
            s1_sig = read_and_resample(s1_source_path)
            s1_gain = float(info_list[1])
            s1_sig = rescale(s1_sig, torch.tensor(len(s1_sig)), s1_gain, scale='dB')
            s2_file = info_list[2].replace('.wv1', '.wav')[len('wsj0/'):]
            s2_source_path = os.path.join(datapath, s2_file)
            s2_sig = read_and_resample(s2_source_path)
            s2_gain = float(info_list[3])
            s2_sig = rescale(s2_sig, torch.tensor(len(s2_sig)), s2_gain, scale='dB')
            # compute mixture
            min_len = min(len(s1_sig), len(s2_sig))
            s1_sig = s1_sig[:min_len]
            s2_sig = s2_sig[:min_len]
            mix_sig = s1_sig + s2_sig
            # rescale, normalize
            sources = torch.stack([s1_sig, s2_sig], dim=0)
            # rescale to avoid clipping
            max_amp = max(
                torch.abs(mix_sig).max().item(),
                *[x.item() for x in torch.abs(sources).max(dim=-1)[0]],
            )
            mix_scaling = 1 / max_amp * 0.9
            sources = mix_scaling * sources
            mix_sig = mix_scaling * mix_sig
            s1_sig = sources[0]
            s2_sig = sources[1]
            # read enrollment
            enr_file = info_list[4].replace('.wv1', '.wav')[len('wsj0/'):]
            enr_source_path = os.path.join(datapath, enr_file)
            enr_sig = read_and_resample(enr_source_path)
            # save to disk
            s1_flag = s1_source_path.split('/')[-1][:-len('.wav')]
            s2_flag = s2_source_path.split('/')[-1][:-len('.wav')]
            enr_flag = enr_source_path.split('/')[-1][:-len('.wav')]
            print(idx)
            s1_out_path = os.path.join(save_dir, 's1', 'idx{:04}_{}_{}_{}.wav'.format(idx, s1_flag, s2_flag, enr_flag))
            s2_out_path = os.path.join(save_dir, 's2', 'idx{:04}_{}_{}_{}.wav'.format(idx, s1_flag, s2_flag, enr_flag))
            enr_out_path = os.path.join(save_dir, 'enr', 'idx{:04}_{}_{}_{}.wav'.format(idx, s1_flag, s2_flag, enr_flag))
            sf.write(s1_out_path, s1_sig.numpy(), target_sr)
            sf.write(s2_out_path, s2_sig.numpy(), target_sr)
            sf.write(enr_out_path, enr_sig.numpy(), target_sr)
            # csv entry
            all_data.append({
                'ID': idx,
                'descriptor': 'idx{:04}_{}_{}_{}'.format(idx, s1_flag, s2_flag, enr_flag),
                's1_path': s1_out_path,
                's2_path': s2_out_path,
                'enr_path': enr_out_path,
                'sr': target_sr,
                'length': len(mix_sig),
                'enr_length': len(enr_sig),
            })
            # next 
            line = f.readline()
            idx += 1
            if limit is not None and idx >= limit:
                break
    df = pd.DataFrame(all_data)
    df.to_csv(os.path.join(save_dir, 'meta.csv'), index=False)


### preparing csv files only
def prepare_dummy_csv(
    savepath,
):
    all_data = [{'ID': i, 'mix_path': '{}.tmp'.format(i)} for i in range(20000)]
    df = pd.DataFrame(all_data)
    df.to_csv(os.path.join(savepath, 'dummy.csv'), index=False)


def create_wsj_csv(datapath, savepath):
    """
    This function creates the csv files to get the speechbrain data loaders for the wsj0-2mix dataset.
    Arguments:
        datapath (str) : path for the wsj0-mix dataset.
        savepath (str) : path where we save the csv file
    """
    import csv
    for set_type in ["cv", "tt"]:
        mix_path = os.path.join(datapath, "wav8k/min/" + set_type + "/mix/")
        s1_path = os.path.join(datapath, "wav8k/min/" + set_type + "/s1/")
        s2_path = os.path.join(datapath, "wav8k/min/" + set_type + "/s2/")

        files = os.listdir(mix_path)

        mix_fl_paths = [mix_path + fl for fl in files]
        s1_fl_paths = [s1_path + fl for fl in files]
        s2_fl_paths = [s2_path + fl for fl in files]

        csv_columns = [
            "ID",
            "mix_path",
            "s1_path",
            "s2_path",
            "enr_path",
            "sr",
            "length",
            "enr_length",
        ]

        with open(savepath + "/wsj_" + set_type + ".csv", "w") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for i, (mix_path, s1_path, s2_path) in enumerate(
                zip(mix_fl_paths, s1_fl_paths, s2_fl_paths)
            ):
                info = torchaudio.info(mix_path)
                row = {
                    "ID": i,
                    "mix_path": mix_path,
                    "s1_path": s1_path,
                    "s2_path": s2_path,
                    "enr_path": s1_path,
                    "sr": info.sample_rate,
                    "length": info.num_frames,
                    "enr_length": info.num_frames,
                }
                writer.writerow(row)


if __name__ == '__main__':
    seed = 123
    hyp = {
        'data_dir': '/mnt/data/Speech/librispeech/dev-clean/dev-clean/',
        'ext': 'flac',
        'save_dir': '/mnt/data/Speech/librispeech_tse/dev-clean-mix/seed{}'.format(seed),
        'limit': 500,
        'dur': 3.,
        'target_sr': 16000,
    }
    random.seed(seed)
    np.random.seed(seed)
    generate_static_data_files(**hyp)

    #  hyp = {
    #      'datapath': '/mnt/data/wsj0.8k/',
    #      'txtpath': '/mnt/data/Speech/wsj_tse/mix_2_spk_tt_extr.txt',
    #      'save_dir': '/mnt/data/Speech/wsj_tse/mix_2_spk_tt',
    #      'limit': None,
    #      'target_sr': 8000,
    #  }
    #  wsj_tse_remix(**hyp)
