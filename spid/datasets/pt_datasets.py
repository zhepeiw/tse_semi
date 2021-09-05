import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import librosa
from tqdm import tqdm
from scipy.signal import resample_poly
import soundfile
import random
import os
import pdb
import time


############################# utility function ###############################
def read_sound(filename, target_sr, start=0, frames=-1):
    sig, sr = soundfile.read(filename, start=start, frames=frames,
                             always_2d=True, dtype='float32')
    sig = sig[:, 0]
    if sr != target_sr:
        sig = resample_poly(sig, target_sr, sr).astype(np.float32)
    return sig


def make_sp_dic(wav_dir, ext):
    wav_paths = sorted(librosa.util.find_files(wav_dir, ext=ext))
    sp_dic = {}
    for path in tqdm(wav_paths):
        spid = str(path.split('/')[-3])
        if spid not in sp_dic:
            sp_dic[spid] = [path]
        else:
            sp_dic[spid].append(path)
    return sp_dic


############################# class ########################################
class SPID_Dataset(Dataset):
    def __init__(self,
                 wav_dir,
                 meta_path,
                 sp_dic_path,
                 ext,
                 target_sr=16000,
                 duration=4.,
                 num_utt=4,
                 ):
        '''
            args:
                wav_dir: directory contains <spid>/<bookid>/wav
        '''
        super().__init__()
        self.sp_dic = make_sp_dic(wav_dir, ext)
        #  if not os.path.exists(sp_dic_path):
        #      self.sp_dic = make_sp_dic(wav_dir, ext)
        #      torch.save(self.sp_dic, sp_dic_path)
        #  else:
        #      self.sp_dic = torch.load(sp_dic_path)

        self.target_sr = target_sr
        self.duration = duration
        self.num_utt = num_utt

        self.spid_list = list(self.sp_dic.keys())


    def __len__(self):
        return len(self.spid_list)

    def get_sound_segment(self, path):
        '''
            get a segment of file and pad to the desired length if necessary

            returns a numpy array of float32
        '''
        info = soundfile.info(path)
        fsr = info.samplerate
        frames = int(fsr * self.duration) if fsr * self.duration < info.frames else -1
        start = np.random.randint(info.frames - frames + 1) if frames != -1 else 0
        sig = read_sound(path, self.target_sr, start, frames)
        segment_length = int(self.duration * self.target_sr)
        if len(sig) < segment_length:
            pad_sig = np.zeros(segment_length).astype(np.float32)
            start = np.random.randint(segment_length - len(sig) + 1)
            pad_sig[start: start + len(sig)] = sig
            sig = pad_sig

        return sig

    def get_dict_item(self, idx):
        idx = idx % len(self.spid_list)
        spid = self.spid_list[idx]
        all_speech = []
        # get random segments
        while len(all_speech) < self.num_utt:
            for attempts in range(10):
                try:
                    filename = random.choice(self.sp_dic[spid])
                    sig = self.get_sound_segment(filename)
                    all_speech.append(sig)
                    break
                except:
                    if attempts == 9:
                        pdb.set_trace()
                    continue
        all_speech = np.hstack(all_speech)[None, :]  # [1, KT]
        dp_dict = {
            'model_inp': all_speech,
        }
        return dp_dict

    def __getitem__(self, idx):
        dp_dict = self.get_dict_item(idx)
        model_inp = torch.from_numpy(dp_dict['model_inp'].astype(np.float32))
        return model_inp


class Combined_Dataset(Dataset):
    def __init__(self, datasets):
        super().__init__()
        self.datasets = datasets
        self.lens = [len(ds) for ds in datasets]
        self.total_len = sum(self.lens)
        print('Total number of speakers: {}'.format(self.total_len))
        self.cum_idxs = np.cumsum(self.lens)

    def __len__(self):
        return self.total_len

    def get_dict_item(self, idx):
        loc = np.nonzero(self.cum_idxs <= idx)[0]
        if len(loc) == 0:
            offset = 0
            ds_idx = 0
        else:
            ds_idx = (loc[-1] + 1).item()
            offset = self.cum_idxs[loc[-1]].item()

        utt_idx = idx - offset
        dp_dict = self.datasets[ds_idx].get_dict_item(utt_idx)
        return dp_dict

    def __getitem__(self, idx):
        dp_dict = self.get_dict_item(idx)
        model_inp = torch.from_numpy(dp_dict['model_inp'].astype(np.float32))
        return model_inp


class OverfitDataset(Dataset):
    def __init__(self, original_ds, num_overfit):
        super().__init__()
        self.overfit_batches = []
        self.original_ds = original_ds
        self.num_overfit = num_overfit

    def __getitem__(self, idx):
        if len(self.overfit_batches) >= self.num_overfit:
            return self.overfit_batches[idx % self.num_overfit]
        else:
            self.overfit_batches.append(self.original_ds[idx])
            return self.overfit_batches[-1]

    def __len__(self):
        return self.num_overfit

    def __getattr__(self, name):
        return getattr(self.original_ds, name)


def create_dataloader(hyp):
    wav_dirs = [dic['wav_dir'] for dic in hyp['datasets/speech']]
    meta_paths = [dic['meta_path'] for dic in hyp['datasets/speech']]
    dic_paths = [dic['sp_dic_path'] for dic in hyp['datasets/speech']]
    exts = [dic['ext'] for dic in hyp['datasets/speech']]

    datasets = [
        SPID_Dataset(wav_dir, meta_path, dic_path, ext,
                    target_sr=hyp['audio/sample_rate'],
                    duration=hyp['datasets/duration'],
                    num_utt=hyp['datasets/num_utterances_per_speaker_per_batch']
                    ) for (wav_dir, meta_path, dic_path, ext) in zip(wav_dirs, meta_paths, dic_paths, exts)
    ]
    comb_ds = Combined_Dataset(datasets)
    if hyp['datasets/num_overfit'] is not None:
        ovf_ds = OverfitDataset(comb_ds, hyp['datasets/num_overfit'])
        train_loader = torch.utils.data.DataLoader(
            ovf_ds,
            batch_size=hyp['train/batch_size'],
            num_workers=hyp['train/num_workers'],
            drop_last=True,
            shuffle=True,
            pin_memory=hyp['train/pin_memory'],
        )
        return ovf_ds, train_loader
    else:
        loader = DataLoader(
            comb_ds,
            batch_size=hyp['train/batch_size'],
            num_workers=hyp['train/num_workers'],
            drop_last=True,
            shuffle=True,
            pin_memory=hyp['train/pin_memory'],
        )
        return comb_ds, loader


def test_single_dataset():
    import time
    wav_dir = '/mnt/data/Speech/VoxCeleb1/dev_wav'
    meta_path = '/mnt/data/Speech/VoxCeleb1/vox1_meta.csv'
    sp_dic_path = '/mnt/data/Speech/VoxCeleb1/sp_dic.pt'
    ext = 'wav'
    ds = SPID_Dataset(wav_dir, meta_path, sp_dic_path, ext)
    train_loader = DataLoader(
        ds,
        batch_size=16,
        num_workers=0,
        drop_last=True,
        shuffle=True,
        pin_memory=False,
    )
    print(len(train_loader))
    tstart = time.time()
    for i, batch in enumerate(train_loader):
        print(batch.shape)
        if i == 10:
            break
    print('time per batch: {}'.format((time.time() - tstart) / 10))


def test_comb_dataset():
    hyp = {
        'datasets/speech': [
            {
                'wav_dir': '/mnt/nvme/Speech/librispeech/train-clean-100/train-clean-100',
                'meta_path': '/mnt/data/Speech/librispeech/train-clean-100/SPEAKERS.TXT',
                'sp_dic_path': '/mnt/data/Speech/librispeech/train-clean-100/sp_dic.pt',
                'ext': 'flac',
            },
            {
                'wav_dir': '/mnt/nvme/Speech/librispeech/train-clean-360/train-clean-360',
                'meta_path': '/mnt/data/Speech/librispeech/train-clean-360/SPEAKERS.TXT',
                'sp_dic_path': '/mnt/data/Speech/librispeech/train-clean-360/sp_dic.pt',
                'ext': 'flac',
            },
            {
                'wav_dir': '/mnt/nvme/Speech/VoxCeleb1/dev_wav',
                'meta_path': '/mnt/data/Speech/VoxCeleb1/vox1_meta.csv',
                'sp_dic_path': '/mnt/data/Speech/VoxCeleb1/sp_dic.pt',
                'ext': 'wav',
            },
            {
                'wav_dir': '/mnt/nvme/Speech/VoxCeleb2/dev/aac',
                'meta_path': '/mnt/data/Speech/VoxCeleb2/vox2_meta.csv',
                'sp_dic_path': '/mnt/data/Speech/VoxCeleb2/sp_dic.pt',
                'ext': 'wav',
            },
        ],
        'audio/sample_rate': 16000,
        'datasets/duration': 3.,
        'datasets/num_utterances_per_speaker_per_batch': 4,
        'datasets/num_overfit': None,
        'train/batch_size': 40,
        'train/num_workers': 16,
        'train/pin_memory': True,
    }
    import time
    train_ds, train_loader = create_dataloader(hyp)
    print(len(train_loader))
    tstart = time.time()
    for i, batch in enumerate(train_loader):
        #  print(batch.shape)
        if i % 10 == 0:
            print('time per batch: {}'.format((time.time() - tstart) / (i+1)))
        if i == 200:
            break


if __name__ == '__main__':
    test_comb_dataset()
