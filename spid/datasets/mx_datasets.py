import audio_base as ab
import base
import data

import traceback
import copy
import numpy as np
import mxnet as mx
import random
import os
import librosa
import json
import logging
from logging import info as loginfo
from logging import debug as logdebug
from scipy import stats
from mxnet import nd
import glob2
import random
import tqdm
import pdb


def _get_random_subsection(samples, size):
    start = np.random.randint(len(samples) - size)
    return samples[start : start + size]


def _make_triplet_dict(ds):

    sp_dict = {}

    for idx in tqdm.tqdm(range(len(ds))):
        _, meta = ds[idx]
        sp_key = str(meta['speaker/id'])
        if sp_key not in sp_dict.keys():
            sp_dict[sp_key] = list()
        sp_dict[sp_key].append(idx)

    return sp_dict


def _crop_if_needed(s, target_len):
    if s.shape[1] > target_len:
        start = np.random.randint(0, s.shape[1] - target_len + 1)
        return s[:, start : start + target_len]
    else:
        return s


class AudioRecordDataset(mx.gluon.data.dataset.RecordFileDataset):
    """ RecordIO dataset that serves audio files

        params
        ------
        filename (str):
            The path of the .rec file
        md_to_tensors (function):
            Function that takes in the decoded metadata dictionary and returns a list of tensors
        audio_size (int):
            length of the audio arrays to be returned
    """

    def __init__(self, filename, md_to_tensors=None, audio_size=16384, expected_sample_rate=16000):
        super(AudioRecordDataset, self).__init__(filename)
        self.md_to_tensors = md_to_tensors
        self.audio_size = audio_size
        self.expected_sample_rate = expected_sample_rate
        self.num_served = 0
        base.util.reseed_random_generators()
        #  self.shift = np.random.randint(len(self))
        self.shift = 0
        self.len = len(self)
        loginfo('dataset len {}, shift {}'.format(len(self), self.shift))

    def __getitem__(self, idx):
        #  if self.num_served <= 10 or self.num_served % self.len == 0:
        #      base.util.reseed_random_generators()
        #      self.shift = np.random.randint(len(self))

        # print(f"serving {idx}")
        record = super(AudioRecordDataset, self).__getitem__((idx + self.shift) % self.len)
        header, samples = mx.recordio.unpack(record)
        # if self._transform is not None:
        #     return self._transform(image.imdecode(img, self._flag), header.label)
        samples = np.frombuffer(samples, dtype=np.float16)

        if self.audio_size < len(samples):
            samples = _get_random_subsection(samples, self.audio_size)
        samples = np.expand_dims(samples, axis=0)
        # metadata about the data-point
        metadata = data.util.decode_floatarr_to_dict(header.label)
        if metadata['audio/sample_rate'] != self.expected_sample_rate:
            samples = ab.util.resample(samples,
                                       metadata['audio/sample_rate'], self.expected_sample_rate)
        ret = samples

        assert self.md_to_tensors is None, "metadata to tensors not supported"
        # if self.md_to_tensors is not None:
            # ret = ret, *self.md_to_tensors(md)

        self.num_served += 1
        return ret, metadata


import torch
class SPID_GE2E_Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 speech_ds,
                 dic_path,
                 ds_name,
                 augmentation_fn=None,
                 transform_fn=None,
                 num_utterances=5,
                 ):
        super().__init__()
        self.speech_ds = speech_ds  # iorecord
        self.ds_name = ds_name
        self.num_utterances = num_utterances

        if not os.path.exists(dic_path):
            print('Generating triplet dictionary to {}'.format(dic_path))
            sp_dic = _make_triplet_dict(speech_ds)
            torch.save(sp_dic, dic_path)
        else:
            print('Loading triplet dictionary from {}'.format(dic_path))
            sp_dic = torch.load(dic_path)
        self.sp_dic = sp_dic
        self.list_of_classes = list(self.sp_dic.keys())
        self.no_classes = len(self.list_of_classes)

        self.augmentation_fn = augmentation_fn
        if self.augmentation_fn is None:
            self.augmentation_fn = lambda _x: _x

        self.transform_fn = transform_fn
        if self.transform_fn is None:
            self.transform_fn = lambda *_x: _x

        self.item_names = list(self.get_dict_item(0).keys())


    def __len__(self):
        #  return len(self.speech_ds)
        return self.no_classes

    def get_dict_item(self, idx):
        idx_anc = idx % self.no_classes
        # Pick speaker based on index
        anchor_speaker_id = self.list_of_classes[idx_anc]
        # Draw utterances of the same speaker at random and append along time
        try:
            rand_idx_list = random.sample(self.sp_dic[anchor_speaker_id], k=self.num_utterances)
        except:
            if len(self.sp_dic[anchor_speaker_id]) == 0:
                pdb.set_trace()
            rand_idx_list = random.choices(self.sp_dic[anchor_speaker_id], k=self.num_utterances)

        for count, idxx in enumerate(rand_idx_list):
            positive_speech, metadata_positive = self.speech_ds[idxx]
            #  print(metadata_positive['speaker/id'])
            #print(idxx)

             # Append
            if count == 0:
                anchor_speech = positive_speech
            else:
                anchor_speech = np.append(anchor_speech, positive_speech, axis=1)

        dp_dict = {'model_inp' : anchor_speech}
        return dp_dict

    def __getitem__(self, idx):
        dp_dict = self.get_dict_item(idx)
        model_inp = torch.from_numpy(dp_dict['model_inp'].astype(np.float32))
        return model_inp


class Combine_SPID_GE2E_Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 datasets,
                 augmentation_fn=None,
                 transform_fn=None,
                ):
        super().__init__()
        self.datasets = datasets  # a list of ge2e dataset objects
        self.lens = [len(x) for x in datasets]
        self.total_len = sum(self.lens)

        self.no_classes = sum([ds.no_classes for ds in datasets])

        self.indices = np.cumsum(self.lens)

        # Transforms and Augmentations
        self.transform_fn = transform_fn
        if self.transform_fn is None:
            self.transform_fn = lambda *_x: _x

        self.augmentation_fn = augmentation_fn
        if self.augmentation_fn is None:
            self.augmentation_fn = lambda _x: _x
        # Dataset tags {'model_inp':}
        self.item_names = self.datasets[0].item_names

    def __len__(self):
        return self.total_len

    def get_dict_item(self, idx):
        loc = np.nonzero(self.indices <= idx)[0]

        if len(loc) == 0:
            offset = 0
            ds_ind = 0
        else:
            ds_ind = (loc[-1] + 1).item()
            offset = self.indices[loc[-1]].item()

        utt_ind = idx - offset

        # Within each dataset, randomly select a speaker and output concatenated speech
        # Random because we have randomized speaker selection in SpeakerID_GE2E_Dataset get_dict_item()
        dp_dict = self.datasets[ds_ind].get_dict_item(utt_ind)

        # Apply augmentations if required (Ideally needs to be applied in SpeakerID_GE2E_Dataset) Modify later accordingly
        dp_dict = self.augmentation_fn(dp_dict)

        # Return
        return dp_dict

    def __getitem__(self, idx):
        dp_dict = self.get_dict_item(idx)
        model_inp = torch.from_numpy(dp_dict['model_inp'].astype(np.float32))
        return model_inp


class OverfitDataset(torch.utils.data.Dataset):
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


def create_dataloader(hyp, part, augmentation_fn=None):
    ds_kwargs = {
        'audio_size': int(hyp['datasets/duration'] * hyp['audio/sample_rate']),
        'expected_sample_rate': hyp['audio/sample_rate'],
    }
    ds_names = [dic['name'] for dic in hyp['datasets/{}/speech'.format(part)]]
    speech_paths = [dic['rec_path'] for dic in hyp['datasets/{}/speech'.format(part)]]
    speech_ds = [AudioRecordDataset(_p, **ds_kwargs) for _p in speech_paths]
    speech_dic_paths = [dic['dic_path'] for dic in hyp['datasets/{}/speech'.format(part)]]

    datasets = [
        SPID_GE2E_Dataset(
            ds,
            dic_path,
            ds_name,
            augmentation_fn=augmentation_fn,
            transform_fn=None,
            num_utterances=hyp['datasets/num_utterances_per_speaker_per_batch'],
        ) for ds, dic_path, ds_name in zip(speech_ds, speech_dic_paths, ds_names)
    ]
    spid_ds = Combine_SPID_GE2E_Dataset(
        datasets,
        augmentation_fn=augmentation_fn,
        transform_fn=None,
    )

    if hyp['datasets/num_overfit'] is not None:
        ovf_ds = OverfitDataset(spid_ds, hyp['datasets/num_overfit'])
        train_loader = torch.utils.data.DataLoader(
            ovf_ds,
            batch_size=hyp['train/batch_size'],
            num_workers=hyp['train/num_workers'],
            drop_last=True,
            shuffle=True
        )
        return train_loader

    if part == 'train':
        train_loader = torch.utils.data.DataLoader(
            spid_ds,
            batch_size=hyp['train/batch_size'],
            num_workers=hyp['train/num_workers'],
            drop_last=True,
            shuffle=True
        )
        return train_loader
    else:
        test_loader = torch.utils.data.DataLoader(
            spid_ds,
            batch_size=1,
            num_workers=0,
            drop_last=True,
            shuffle=False
        )
        return test_loader


############################ test cases ######################################
def test_spid_loader():
    hyp = {
        'datasets/train/speech': [
            {
                'name': 'vc1',
                'rec_path': '/home/ubuntu/local_training_cache/vc1_orig_all_recordio_gender_4secs_2020_08_26/000000.rec',
                'dic_path': '/home/ubuntu/local_training_cache/vc1_orig_all_recordio_gender_4secs_2020_08_26/sp_dic.pt',
            },
            {
                'name': 'vc2',
                'rec_path': '/home/ubuntu/local_training_cache/vc2_orig_all_recordio_gender_6secs_2021_06_25_fixed/000000.rec',
                'dic_path': '/home/ubuntu/local_training_cache/vc2_orig_all_recordio_gender_6secs_2021_06_25_fixed/sp_dic.pt',
            },
        ],
        'train/batch_size': 16,
        'train/num_workers': 0,

        'datasets/duration': 64000/16000,
        'datasets/num_utterances_per_speaker_per_batch': 4,
        'datasets/num_overfit': None,
        'audio/sample_rate': 16000,
    }

    augmentation_fn = None
    train_loader = create_dataloader(hyp, 'train', augmentation_fn)
    for i, batch in enumerate(train_loader):
        print(batch.shape)
        if i == 20: break


def test_spid_ovf_loader():
    hyp = {
        'datasets/train/speech': [
            {
                'name': 'vc1',
                'rec_path': '/home/ubuntu/local_training_cache/vc1_orig_all_recordio_gender_4secs_2020_08_26/000000.rec',
                'dic_path': '/home/ubuntu/local_training_cache/vc1_orig_all_recordio_gender_4secs_2020_08_26/sp_dic.pt',
            },
            {
                'name': 'vc2',
                'rec_path': '/home/ubuntu/local_training_cache/vc2_orig_all_recordio_gender_6secs_2021_06_25_fixed/000000.rec',
                'dic_path': '/home/ubuntu/local_training_cache/vc2_orig_all_recordio_gender_6secs_2021_06_25_fixed/sp_dic.pt',
            },
        ],
        'train/batch_size': 4,
        'train/num_workers': 0,

        'datasets/duration': 64000/16000,
        'datasets/num_utterances_per_speaker_per_batch': 4,
        'datasets/num_overfit': 8,
        'audio/sample_rate': 16000,
    }

    augmentation_fn = None
    train_loader = create_dataloader(hyp, 'train', augmentation_fn)
    for i, batch in enumerate(train_loader):
        print(batch.shape)
        if i == 20: break


if __name__ == '__main__':
    test_spid_ovf_loader()
