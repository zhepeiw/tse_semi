import torch
import librosa
import pandas as pd
from tqdm import tqdm
import soundfile as sf
import numpy as np
import os
import pdb
import argparse


def split_segments_idxs(total_length, seg_length):
    '''
        returns a list of tuples (start, end) with equal length
        except for the last segment
    '''
    idxs = []
    start = 0
    while start < total_length:
        end = min(total_length, start + seg_length)
        idxs.append((start, end))
        if end == total_length:
            break
        start = end
    return idxs


def sp_to_gender_dict(csv_path):
    if 'vox1' in csv_path:
        df = pd.read_csv(csv_path, sep='\t')
    else:
        df = pd.read_csv(csv_path)
    sp_dic = {}
    for idx in range(len(df)):
        if 'vox1' in csv_path:
            spid = str(df['VoxCeleb1 ID'][idx]).strip()
            gender = df['Gender'][idx].strip()
        else:
            spid = str(df['VoxCeleb2 ID '][idx]).strip()
            gender = df['Gender '][idx].strip()
        if gender == 'm':
            sp_dic[spid] = 'M'
        else:
            sp_dic[spid] = 'F'
    return sp_dic


def process_dataset(hyp, debug=False):
    wav_paths = librosa.util.find_files(hyp['wav_dir'], ext=hyp['ext'])
    if debug:
        wav_paths = wav_paths[:20]
    batch_paths = [wav_paths[i::hyp['num_workers']] for i in range(hyp['num_workers'])]

    os.makedirs(hyp['pt_dir'], exist_ok=True)
    if hyp['num_workers'] == 1:
        cnt = process_vc_file(batch_paths[0])
    else:
        import multiprocessing
        p = multiprocessing.Pool(hyp['num_workers'])
        cnt_list = p.map(process_vc_file, batch_paths)
        pdb.set_trace()
        cnt = sum(cnt_list)
    print('Processed {} pt files'.format(cnt))


def process_vc_file(wav_paths):
    hyp = global_hyp
    gender_dict = sp_to_gender_dict(hyp['meta_path'])
    cnt = 0
    for path in tqdm(wav_paths):
        try:
            samples, sr = sf.read(path)
            assert len(samples.shape) == 1
            if sr != hyp['target_sr']:
                samples = librosa.resample(samples, sr, hyp['target_sr'])
                sr = hyp['target_sr']
            samples = samples.astype(np.float32)
            path_splits = path.split('/')
            spid = path_splits[-3]
            file_id = '_'.join(path_splits[-3:])[:-len('.wav')]
        except:
            continue

        # split into segments
        segment_length = int(hyp['target_sr'] * hyp['segment_sec'])
        segment_idxs = split_segments_idxs(len(samples), segment_length)
        for start, end in segment_idxs:
            if end - start == segment_length:
                chunk = samples[start:end]
            elif end - start >= 0.5 * segment_length:
                chunk = np.zeros(segment_length).astype(np.float32)
                tmp = np.random.randint(segment_length - (end - start) + 1)
                chunk[tmp: tmp + (end - start)] = samples[start:end]
            else:
                continue
            # <spid>_<passage>_<file>_<start>_<end>.pt
            pt_path = os.path.join(hyp['pt_dir'], '{}_{}_{}.pt'.format(
                file_id, start, end
            ))
            info = {
                'data': torch.from_numpy(chunk),
                'source_path': path,
                'spid': spid,
                'gender': gender_dict[spid],
            }
            torch.save(info, pt_path)
            cnt += 1
    return cnt


VC1_HYP = {
    'wav_dir': '/mnt/data/Speech/VoxCeleb1/dev_wav',
    'meta_path': '/mnt/data/Speech/VoxCeleb1/vox1_meta.csv',
    'pt_dir': '/mnt/data/Speech/preprocessed_zhepei/VoxCeleb1/dev_pts',
    'ext': 'wav',
    'num_workers': 20,
    'target_sr': 16000,
    'segment_sec': 8.,
}

global_hyp = VC1_HYP

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    hyp = global_hyp
    assert hyp['num_workers'] > 0
    process_dataset(hyp, debug=args.debug)
