hyp = {

    'log/base_dir': '/mnt/nvme/zhepei/outputs/tse_semi/spid',
    'log/chkpt_dir': 'chkpt',
    'log/use_wandb': True,
    'log/wandb/project': 'spid',
    'log/wandb/entity': 'CAL',

    'pretrain': False,

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
    'datasets/duration': 4.,
    'datasets/num_utterances_per_speaker_per_batch': 4,
    'datasets/num_overfit': None,

    'embedder/n_fft': 512,
    'embedder/hop': 160,
    'embedder/n_mels': 40,
    'embedder/lstm_hidden': 768,
    'embedder/num_layers': 3,
    'embedder/emb_dim': 256,

    'train/batch_size': 16,
    'train/num_workers': 16,
    'train/pin_memory': True,
    'train/optimizer': 'adam',
    'train/adam': 5e-4,
    'train/scheduler/milestones': [50000, 100000, 150000, 200000, 250000, 300000],
    'train/scheduler/gamma': 0.5,
    'train/clip_grad_norm': 3.0,
    'train/summary_interval': 30,
    'train/checkpoint_interval': 25000,

    'eval/eval_interval': 25000,
    'eval/path_to_test_list': '/mnt/data/Speech/VoxCeleb1/veri_test.txt',
    'eval/path_to_test': '/mnt/data/Speech/VoxCeleb1/test_wav',
    'eval/num_files': None,
}


def blstm_softmax(experiment_name='vfpt_spid', run_name='blstm_softmax'):
    hyp_update_dict = {

    }
    hyp.update(hyp_update_dict)
    return experiment_name, run_name, hyp


def blstm_softmax_ovf(experiment_name='vfpt_spid', run_name='blstm_softmax_ovf'):
    hyp_update_dict = {
        'datasets/num_overfit': 80,
        'train/batch_size': 16,
        'train/pin_memory': False,
        'train/num_workers': 16,
        'train/adam': 5e-5,
        'train/scheduler/milestones': [500, 1000, 5000, 10000],
        'train/scheduler/gamma': 0.5,
        'train/checkpoint_interval': 1000000,

        'eval/eval_interval': 1000000,
        'eval/num_files': 10,
    }
    hyp.update(hyp_update_dict)
    return experiment_name, run_name, hyp


def blstm_softmax_bs40_3sec(experiment_name='vfpt_spid', run_name='blstm_softmax_bs40_3sec'):
    hyp_update_dict = {
        'datasets/duration': 3.,
        'train/batch_size': 40,
        'train/num_workers': 16,
        'train/pin_memory': True,
        'train/adam': 1e-4,
        'train/scheduler/milestones': [50000, 100000, 150000, 200000, 250000, 300000],
        'train/scheduler/gamma': 0.5,

        'pretrain': True,
        'pretrain/ckpt_path': '/mnt/nvme/zhepei/outputs/tse_semi/spid/vfpt_spid/2021-08-17-21-58-03_blstm_softmax_bs40_3sec/chkpt/chkpt_50000.pt',
    }
    hyp.update(hyp_update_dict)
    return experiment_name, run_name, hyp


def blstm_softmax_bs40_3sec_ovf(experiment_name='vfpt_spid', run_name='blstm_softmax_bs40_3sec_ovf'):
    hyp_update_dict = {
        'datasets/num_overfit': 80,
        'datasets/duration': 3.,
        'train/batch_size': 40,
        'train/num_workers': 16,
        'train/pin_memory': True,
        'train/adam': 1e-4,
        'train/scheduler/milestones': [500, 1000, 5000, 10000],
        'train/scheduler/gamma': 0.5,
        'train/checkpoint_interval': 1000000,

        'eval/eval_interval': 1000000,
        'eval/num_files': 10,
    }
    hyp.update(hyp_update_dict)
    return experiment_name, run_name, hyp


def blstm_softmax_bs46_3sec(experiment_name='vfpt_spid', run_name='blstm_softmax_bs46_3sec'):
    hyp_update_dict = {
        'datasets/duration': 3.2,
        'train/batch_size': 46,
        'train/num_workers': 16,
        'train/pin_memory': True,
        'train/adam': 1e-4,
        'train/scheduler/milestones': [50000, 100000, 150000, 200000, 250000, 300000],
        'train/scheduler/gamma': 0.5,
    }
    hyp.update(hyp_update_dict)
    return experiment_name, run_name, hyp


def blstm_softmax_8k_bs46_4sec(experiment_name='vfpt_spid', run_name='blstm_softmax_8k_bs46_4sec'):
    hyp_update_dict = {
        'audio/sample_rate': 8000,
        'datasets/duration': 4,
        'train/batch_size': 48,
        'train/num_workers': 16,
        'train/pin_memory': True,
        'train/adam': 1e-4,
        'train/scheduler/milestones': [50000, 100000, 150000, 200000, 250000, 300000],
        'train/scheduler/gamma': 0.5,
    }
    hyp.update(hyp_update_dict)
    return experiment_name, run_name, hyp


