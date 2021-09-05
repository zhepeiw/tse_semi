hyp = {
    'log/base_dir': '/home/ubuntu/efs/transient/wangzhep',
    'log/chkpt_dir': 'chkpt',

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
    'datasets/duration': 64000/16000,
    'datasets/num_utterances_per_speaker_per_batch': 4,
    'datasets/num_overfit': None,

    'audio/sample_rate': 16000,

    'embedder/n_fft': 512,
    'embedder/hop': 160,
    'embedder/n_mels': 40,
    'embedder/lstm_hidden': 768,
    'embedder/num_layers': 3,
    'embedder/emb_dim': 256,

    'train/batch_size': 16,
    'train/num_workers': 8,
    'train/optimizer': 'adam',
    'train/adam': 5e-6,
    'train/scheduler/milestones': [2000, 5000, 10000],
    'train/scheduler/gamma': 0.6,
    'train/clip_grad_norm': None,
    'train/summary_interval': 30,
    'train/checkpoint_interval': 25000,

    'eval/eval_interval': 25000,
    'eval/path_to_test_list': '/home/ubuntu/local_training_cache/spid/voxceleb1_veri_test.txt',
    'eval/path_to_test': '/home/ubuntu/local_training_cache/spid/wav',
    'eval/num_files': None,
}


def blstm_softmax(experiment_name='vfpt_spid', run_name='blstm_softmax'):
    hyp_update_dict = {
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
            {
                'name': 'libri',
                'rec_path': '/home/ubuntu/local_training_cache/libri_train_clean_100_recordio_10secs_2021_06_02/train/000000.rec',
                'dic_path': '/home/ubuntu/local_training_cache/libri_train_clean_100_recordio_10secs_2021_06_02/train/sp_dic.pt',
            },
            {
                'name': 'libri',
                'rec_path': '/home/ubuntu/local_training_cache/libri_train_clean_360_recordio_10secs_2021_06_02/train/000000.rec',
                'dic_path': '/home/ubuntu/local_training_cache/libri_train_clean_360_recordio_10secs_2021_06_02/train/sp_dic.pt',
            },
        ],

        'train/batch_size': 16,
        'datasets/num_utterances_per_speaker_per_batch': 4,
        'train/num_workers': 16,
        'train/adam': 5e-4,
        #  'train/optimizer': 'sgd',
        #  'train/sgd': 0.01,
        'train/scheduler/milestones': [50000, 100000, 150000, 200000, 250000, 300000],
        'train/scheduler/gamma': 0.5,
        'train/clip_grad_norm': 3.0,
    }
    hyp.update(hyp_update_dict)
    return experiment_name, run_name, hyp


def blstm_softmax_bs48(experiment_name='vfpt_spid', run_name='blstm_softmax_bs48'):
    hyp_update_dict = {
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
            {
                'name': 'libri',
                'rec_path': '/home/ubuntu/local_training_cache/libri_train_clean_100_recordio_10secs_2021_06_02/train/000000.rec',
                'dic_path': '/home/ubuntu/local_training_cache/libri_train_clean_100_recordio_10secs_2021_06_02/train/sp_dic.pt',
            },
            {
                'name': 'libri',
                'rec_path': '/home/ubuntu/local_training_cache/libri_train_clean_360_recordio_10secs_2021_06_02/train/000000.rec',
                'dic_path': '/home/ubuntu/local_training_cache/libri_train_clean_360_recordio_10secs_2021_06_02/train/sp_dic.pt',
            },
        ],

        'train/batch_size': 48,
        'datasets/num_utterances_per_speaker_per_batch': 4,
        'train/num_workers': 8,
        'train/adam': 1e-4,
        #  'train/optimizer': 'sgd',
        #  'train/sgd': 0.01,
        'train/scheduler/milestones': [50000, 100000, 150000, 200000, 250000, 300000],
        'train/scheduler/gamma': 0.5,
        'train/clip_grad_norm': 3.0,
    }
    hyp.update(hyp_update_dict)
    return experiment_name, run_name, hyp


def blstm_softmax_ovf(experiment_name='vfpt_spid', run_name='blstm_softmax_ovf'):
    hyp_update_dict = {
        'datasets/num_overfit': 80,
        'train/batch_size': 16,
        'train/num_workers': 8,
        'train/adam': 5e-5,
        'train/scheduler/milestones': [500, 1000, 5000, 10000],
        'train/scheduler/gamma': 0.5,
        'train/checkpoint_interval': 1000000,

        'eval/eval_interval': 1000000,
        'eval/num_files': 10,
    }
    hyp.update(hyp_update_dict)
    return experiment_name, run_name, hyp


def reproduce(experiment_name='vfpt_spid', run_name='reproduce'):
    hyp_update_dict = {
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
            {
                'name': 'libri',
                'rec_path': '/home/ubuntu/local_training_cache/libri_train_clean_100_recordio_10secs_2021_06_02/train/000000.rec',
                'dic_path': '/home/ubuntu/local_training_cache/libri_train_clean_100_recordio_10secs_2021_06_02/train/sp_dic.pt',
            },
            {
                'name': 'libri',
                'rec_path': '/home/ubuntu/local_training_cache/libri_train_clean_360_recordio_10secs_2021_06_02/train/000000.rec',
                'dic_path': '/home/ubuntu/local_training_cache/libri_train_clean_360_recordio_10secs_2021_06_02/train/sp_dic.pt',
            },
        ],

        'train/batch_size': 4,
        'datasets/num_utterances_per_speaker_per_batch': 5,
        'train/num_workers': 4,
        'train/optimizer': 'sgd',
        'train/sgd': 0.01,
        'train/scheduler/milestones': [999999],
        'train/scheduler/gamma': 0.99999,
        'train/clip_grad_norm': 3.0,
        'train/summary_interval': 30,
    }
    hyp.update(hyp_update_dict)
    return experiment_name, run_name, hyp
