import torch
import torch.nn as nn
try:
    import modules.torch_stft as torch_stft
except:
    import torch_stft
import numpy as np
import librosa
import pdb


class Wav_to_Mel_VF(nn.Module):
    def __init__(self, sr=16000, n_fft=1024, hop=256, n_mels=40,
                 trainable=False):
        super().__init__()
        self.stft = torch_stft.STFT(filter_length=n_fft, hop_length=hop)
        n_dim = n_fft // 2 + 1
        self.mel_transform = nn.Linear(n_dim, n_mels, bias=False)
        mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft,
                                        n_mels=n_mels).astype(np.float32)
        mel_basis = torch.from_numpy(mel_basis)
        self.mel_transform.weight = nn.Parameter(mel_basis)

        for p in self.parameters():
            p.requires_grad = trainable


    def forward(self, x):
        '''
            args:
                x: [bs, 1, t]
        '''
        if x.dim() == 2:
            x = x.unsqueeze(1)
        mag, _ = self.stft.transform(x)  # [B, F, T]
        mag = mag ** 2
        mag = mag.permute(0, 2, 1).contiguous() # [B, T, F]
        mel = self.mel_transform(mag)  # [B, T, M]
        mel = mel.permute(0, 2, 1).contiguous()  # [B, M, T]
        mel = torch.log10(mel + 1e-6)
        return mel


class Embedder(nn.Module):
    def __init__(self, hyp):
        super().__init__()
        self.hyp = hyp
        self.mel_transform = Wav_to_Mel_VF(
            sr=hyp['audio/sample_rate'],
            n_fft=hyp['embedder/n_fft'],
            hop=hyp['embedder/hop'],
            n_mels=hyp['embedder/n_mels'],
            trainable=False,
        )

        self.lstm = nn.LSTM(
            hyp['embedder/n_mels'],
            hyp['embedder/lstm_hidden'],
            num_layers=hyp['embedder/num_layers'],
            bidirectional=True,
            batch_first=True,
        )

        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)

        self.fc = nn.Linear(2*hyp['embedder/lstm_hidden'],
                            hyp['embedder/emb_dim'])

    def forward(self, x):
        '''
            args:
                x: [B, L] or [B, 1, L]
        '''
        mel = self.mel_transform(x)  # [B, M, T]
        mel = mel.permute(0, 2, 1).contiguous()  # [B, T, M]
        hidd, _ = self.lstm(mel)  # [B, T, H]
        hidd = (hidd[:, 0] + hidd[:, -1]) / 2  # [B, H]
        #  hidd = hidd[:, -1]  # [B, H]
        out = self.fc(hidd)  # [B, D]
        out = out / torch.norm(out, p=2, dim=1, keepdim=True)  # [B, D]
        return out

############### test cases ################
def test_mel():
    torch.manual_seed(0)
    inp = torch.randn(4, 48000)
    model = Wav_to_Mel_VF(
        sr=16000, n_fft=512, hop=160, n_mels=40, trainable=False
    )
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    tmp = model(inp)
    pdb.set_trace()


def test_emb():
    hyp = {
        'audio/sample_rate': 16000,
        'embedder/n_fft': 512,
        'embedder/hop': 160,
        'embedder/n_mels': 40,
        'embedder/lstm_hidden': 768,
        'embedder/num_layers': 3,
        'embedder/emb_dim': 256,
    }
    torch.manual_seed(0)
    device = torch.device('cpu')
    model = Embedder(hyp).to(device)
    inp = torch.randn(64, 1, 64000).to(device)
    out = model(inp)
    pdb.set_trace()


if __name__ == '__main__':
    #  test_mel()
    test_emb()

