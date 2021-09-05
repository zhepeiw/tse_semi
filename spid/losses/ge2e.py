import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

### adapted from https://github.com/HarryVolek/PyTorch_Speaker_Verification/blob/master/utils.py ###

def get_centroids(embeddings):
    centroids = embeddings.mean(dim=1)
    return centroids


def get_utterance_centroids(embeddings):
    """
    Returns the centroids for each utterance of a speaker, where
    the utterance centroid is the speaker centroid without considering
    this utterance
    Shape of embeddings should be:
        (speaker_ct, utterance_per_speaker_ct, embedding_size)
    """
    sum_centroids = embeddings.sum(dim=1)
    # we want to subtract out each utterance, prior to calculating the
    # the utterance centroid
    sum_centroids = sum_centroids.reshape(
        sum_centroids.shape[0], 1, sum_centroids.shape[-1]
    )
    # we want the mean but not including the utterance itself, so -1
    num_utterances = embeddings.shape[1] - 1
    centroids = (sum_centroids - embeddings) / num_utterances
    return centroids


def get_cossim(embeddings, centroids):
    # number of utterances per speaker
    num_utterances = embeddings.shape[1]
    utterance_centroids = get_utterance_centroids(embeddings)

    # flatten the embeddings and utterance centroids to just utterance,
    # so we can do cosine similarity
    utterance_centroids_flat = utterance_centroids.view(
        utterance_centroids.shape[0] * utterance_centroids.shape[1],
        -1
    )
    embeddings_flat = embeddings.view(
        embeddings.shape[0] * num_utterances,
        -1
    )
    # the cosine distance between utterance and the associated centroids
    # for that utterance
    # this is each speaker's utterances against his own centroid, but each
    # comparison centroid has the current utterance removed
    cos_same = F.cosine_similarity(embeddings_flat, utterance_centroids_flat)

    # now we get the cosine distance between each utterance and the other speakers'
    # centroids
    # to do so requires comparing each utterance to each centroid. To keep the
    # operation fast, we vectorize by using matrices L (embeddings) and
    # R (centroids) where L has each utterance repeated sequentially for all
    # comparisons and R has the entire centroids frame repeated for each utterance
    centroids_expand = centroids.repeat((num_utterances * embeddings.shape[0], 1))
    embeddings_expand = embeddings_flat.unsqueeze(1).repeat(1, embeddings.shape[0], 1)
    embeddings_expand = embeddings_expand.view(
        embeddings_expand.shape[0] * embeddings_expand.shape[1],
        embeddings_expand.shape[-1]
    )
    cos_diff = F.cosine_similarity(embeddings_expand, centroids_expand)
    cos_diff = cos_diff.view(
        embeddings.size(0),
        num_utterances,
        centroids.size(0)
    )
    # assign the cosine distance for same speakers to the proper idx
    same_idx = list(range(embeddings.size(0)))
    cos_diff[same_idx, :, same_idx] = cos_same.view(embeddings.shape[0], num_utterances)
    cos_diff = cos_diff + 1e-6
    return cos_diff


def calc_loss_softmax(sim_matrix):
    same_idx = list(range(sim_matrix.size(0)))
    pos = sim_matrix[same_idx, :, same_idx]
    neg = (torch.exp(sim_matrix).sum(dim=2) + 1e-6).log_()
    per_embedding_loss = -1 * (pos - neg)
    loss = per_embedding_loss.sum()
    return loss, per_embedding_loss


def calc_loss_contrast(sim_matrix):
    def skip_range(i):
        idxs = [*range(N)]
        idxs.remove(i)
        return idxs
    N, M, _ = sim_matrix.shape
    same_idx = list(range(N))
    pos = sim_matrix[same_idx, :, same_idx]
    pos_loss = 1 - torch.sigmoid(pos)
    sig_neg_sim = torch.sigmoid(sim_matrix)

    neg_loss = torch.stack([sig_neg_sim[i, :, skip_range(i)] for i in range(N)], dim=0)  # [N, M, N]
    neg_loss = torch.max(neg_loss, dim=-1)[0]
    per_embedding_loss = pos_loss + neg_loss
    loss = per_embedding_loss.sum()
    return loss, per_embedding_loss


class GE2ELoss(nn.Module):
    def __init__(self, loss_type='softmax'):
        super(GE2ELoss, self).__init__()
        assert loss_type in ['softmax', 'contrast']
        self.w = nn.Parameter(torch.tensor(10.0), requires_grad=True)
        self.b = nn.Parameter(torch.tensor(-5.0), requires_grad=True)
        self.loss_fn = calc_loss_softmax if loss_type == 'softmax' else calc_loss_contrast

    def forward(self, embeddings):
        torch.clamp(self.w, 1e-6)
        centroids = get_centroids(embeddings)
        cossim = get_cossim(embeddings, centroids)
        sim_matrix = self.w*cossim.to(embeddings.device) + self.b
        loss, _ = self.loss_fn(sim_matrix)
        return loss


############################## test fn ###################################
def test_ge2e_softmax():
    torch.manual_seed(0)
    device = torch.device('cpu')
    inp = torch.randn(16, 4, 256).to(device)
    loss_fn = GE2ELoss().to(device)
    loss = loss_fn(inp)
    pdb.set_trace()


if __name__ == '__main__':
    test_ge2e_softmax()
