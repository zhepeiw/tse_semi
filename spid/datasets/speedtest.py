import torch
import soundfile
import os
import glob
import librosa
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pdb


def read_sound(filename, target_sr, start=0, frames=-1):
    sig, sr = soundfile.read(filename, start=start, frames=frames,
                             always_2d=True, dtype='float32')
    sig = sig[:, 0]
    if sr != target_sr:
        sig = resample_poly(sig, target_sr, sr)
    return sig


class SpeedTestDataset(Dataset):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        if mode == 'pt':
            self.pt_list = glob.glob(os.path.join('/mnt/data/Speech/preprocessed_zhepei/VoxCeleb1/dev_pts', '*.pt'))
        elif mode == 'soundfile':
            self.wav_list = sorted(librosa.util.find_files('/mnt/data/Speech/VoxCeleb1/dev_wav', ext='wav'))
        self.duration = 3
        self.target_sr = 16000

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


    def __len__(self):
        return 100

    def __getitem__(self, idx):
        if self.mode == 'pt':
            full_data = torch.load(self.pt_list[idx])['data']
            segment_length = int(self.duration * self.target_sr)
            start = torch.randint(len(full_data) - segment_length + 1, (1,)).item()
            data = full_data[start: start + segment_length]
        elif self.mode == 'soundfile':
            data = self.get_sound_segment(self.wav_list[idx])
            data = torch.from_numpy(data)

        return data


def test_pt():
    import time
    ds = SpeedTestDataset('soundfile')
    dl = DataLoader(
        ds,
        batch_size=8,
        num_workers=0,
        drop_last=True,
        shuffle=True,
        pin_memory=True,
    )
    t_start = time.time()
    cnt = 0
    for batch in dl:
        print(batch.shape)
    print('avg time: {}'.format((time.time()-t_start)/len(dl)))


if __name__ == '__main__':
    test_pt()
