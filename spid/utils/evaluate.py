from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
import numpy as np
import torch
import os
import tqdm
import soundfile
from scipy.signal import resample_poly
import pdb


def compute_roc_EER(y, y_score):
    '''
        adapted from https://stackoverflow.com/questions/28339746/equal-error-rate-in-python
        args:
            y: [bs,], binary
            y_score: [bs,], between 0 and 1


    '''

    fpr, tpr, thresholds = roc_curve(y, y_score, pos_label=1)

    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)
    return eer, thresh


def _get_random_subsection(samples, size):
    start = np.random.randint(len(samples) - size)
    return samples[start : start + size]


def evaluate_verification(model, hyp, input_size=48000):
    device = next(model.parameters()).device
    model.eval()

    path_to_test_list = hyp['eval/path_to_test_list']
    path_to_test = hyp['eval/path_to_test']

    verify_list = np.loadtxt(path_to_test_list, str)
    if hyp['eval/num_files'] is not None:
        verify_list = verify_list[:hyp['eval/num_files']]

    verify_lb = np.array([int(i[0]) for i in verify_list])
    list1 = np.array([os.path.join(path_to_test, i[1]) for i in verify_list])
    list2 = np.array([os.path.join(path_to_test, i[2]) for i in verify_list])

    total_list = np.concatenate((list1, list2))
    unique_list = np.unique(total_list)

    feats, scores, labels = [], [], []
    with torch.no_grad():
        for c, fpath in tqdm.tqdm(enumerate(unique_list)):
            #if c % 50 == 0: print('Finish extracting features for {}/{}th wav.'.format(c, total_length))
            samples, sr = soundfile.read(fpath, dtype='float32', always_2d=True)
            samples = samples[:, 0]
            if sr != hyp['audio/sample_rate']:
                samples = resample_poly(samples, hyp['audio/sample_rate'], sr).astype(np.float32)

            for n_seg in range(3):
                if samples.shape[0] > input_size:
                    samples_seg = _get_random_subsection(samples, input_size)
                else:
                    samples_seg = samples
                samples_seg = samples_seg.reshape(1, -1)
                samples_seg = torch.from_numpy(samples_seg.astype(np.float32)).to(device)

                emb = model(samples_seg.unsqueeze(0))

                if n_seg == 0:
                    emb_tot = emb
                else:
                    emb_tot += emb
            feats += [emb_tot.cpu().numpy()/(n_seg + 1)]

    feats = np.array(feats)

    # ==> compute the pair-wise similarity.
    for c, (p1, p2) in enumerate(zip(list1, list2)):
        ind1 = np.where(unique_list == p1)[0][0]
        ind2 = np.where(unique_list == p2)[0][0]

        v1 = feats[ind1, 0]
        v1 = v1 / (np.linalg.norm(v1, ord=2) + 1e-5)
        v2 = feats[ind2, 0]
        v2 = v2 / (np.linalg.norm(v2, ord=2) + 1e-5)

        scores += [np.sum(v1*v2)]
        labels += [verify_lb[c]]
        #print("Speaker 1: " + str(p1))
        #print("Speaker 2: " + str(p2))
        print('scores : {}, gt : {}'.format(scores[-1], verify_lb[c]))

    scores = np.array(scores)
    labels = np.array(labels)

    eer, thresh = compute_roc_EER(labels, scores)
    print('==> EER: {}'.format(eer))
    model.train()
    return eer
