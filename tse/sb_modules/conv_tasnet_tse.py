import torch
import torch.nn as nn
import speechbrain as sb
import torch.nn.functional as F
from speechbrain.lobes.models.conv_tasnet import Chomp1d, choose_norm, ChannelwiseLayerNorm, GlobalLayerNorm
import pdb


class MaskNet(nn.Module):
    """
    Arguments
    ---------
    N : int
        Number of filters in autoencoder.
    B : int
        Number of channels in bottleneck 1 × 1-conv block.
    H : int
        Number of channels in convolutional blocks.
    P : int
        Kernel size in convolutional blocks.
    X : int
        Number of convolutional blocks in each repeat.
    R : int
        Number of repeats.
    C : int
        Number of speakers.
    emb_dim: int
        Dimension of the embedding
    norm_type : str
        One of BN, gLN, cLN.
    causal : bool
        Causal or non-causal.
    mask_nonlinear : str
        Use which non-linear function to generate mask, in ['softmax', 'relu'].
    Example:
    ---------
    >>> N, B, H, P, X, R, C = 11, 12, 2, 5, 3, 1, 2
    >>> MaskNet = MaskNet(N, B, H, P, X, R, C)
    >>> mixture_w = torch.randn(10, 11, 100)
    >>> est_mask = MaskNet(mixture_w)
    >>> est_mask.shape
    torch.Size([2, 10, 11, 100])
    """

    def __init__(
        self,
        N,
        B,
        H,
        P,
        X,
        R,
        C,
        emb_dim,
        norm_type="gLN",
        causal=False,
        fusion_type='cat',
        mask_nonlinear="relu",
    ):
        super(MaskNet, self).__init__()

        # Hyper-parameter
        self.C = C
        self.mask_nonlinear = mask_nonlinear

        # Components
        # [M, K, N] -> [M, K, N]
        self.layer_norm = ChannelwiseLayerNorm(N)

        # [M, K, N] -> [M, K, B]
        self.bottleneck_conv1x1 = sb.nnet.CNN.Conv1d(
            in_channels=N, out_channels=B, kernel_size=1, bias=False,
        )

        # [M, K, B] -> [M, K, B]
        in_shape = (None, None, B)
        self.temporal_conv_net = TemporalBlocksSequential(
            in_shape, emb_dim, H, P, R, X, norm_type, causal, fusion_type
        )

        # [M, K, B] -> [M, K, C*N]
        self.mask_conv1x1 = sb.nnet.CNN.Conv1d(
            in_channels=B, out_channels=C * N, kernel_size=1, bias=False
        )

    def forward(self, mixture_w, emb):
        """Keep this API same with TasNet.
        Arguments
        ---------
        mixture_w : Tensor
            Tensor shape is [M, K, N], M is batch size.
        emb: Tensor
            Tensor shape [M, D]
        Returns
        -------
        est_mask : Tensor
            Tensor shape is [M, K, C, N].
        """
        mixture_w = mixture_w.permute(0, 2, 1)
        M, K, N = mixture_w.size()
        y = self.layer_norm(mixture_w)
        y = self.bottleneck_conv1x1(y)
        y = self.temporal_conv_net(y, emb)
        score = self.mask_conv1x1(y)

        # score = self.network(mixture_w)  # [M, K, N] -> [M, K, C*N]
        score = score.contiguous().reshape(
            M, K, self.C, N
        )  # [M, K, C*N] -> [M, K, C, N]

        # [M, K, C, N] -> [C, M, N, K]
        score = score.permute(2, 0, 3, 1)

        if self.mask_nonlinear == "softmax":
            est_mask = F.softmax(score, dim=2)
        elif self.mask_nonlinear == "relu":
            est_mask = F.relu(score)
        else:
            raise ValueError("Unsupported mask non-linear function")
        return est_mask


class TemporalBlocksSequential(nn.Module):
    """
    A wrapper for the temporal-block layer to replicate it
    Arguments
    ---------
    input_shape : tuple
        Expected shape of the input.
    emb_dim: int
        Dimension of the embedding
    H : int
        The number of intermediate channels.
    P : int
        The kernel size in the convolutions.
    R : int
        The number of times to replicate the multilayer Temporal Blocks.
    X : int
        The number of layers of Temporal Blocks with different dilations.
    norm type : str
        The type of normalization, in ['gLN', 'cLN'].
    causal : bool
        To use causal or non-causal convolutions, in [True, False].
    Example
    -------
    >>> x = torch.randn(14, 100, 10)
    >>> H, P, R, X = 10, 5, 2, 3
    >>> TemporalBlocks = TemporalBlocksSequential(
    ...     x.shape, H, P, R, X, 'gLN', False, 'cat'
    ... )
    >>> y = TemporalBlocks(x)
    >>> y.shape
    torch.Size([14, 100, 10])
    """

    def __init__(self, input_shape, emb_dim, H, P, R, X, norm_type, causal, fusion_type):
        super().__init__()
        B = input_shape[-1]
        self.fusion_modules = nn.ModuleList([])
        self.ctn_modules = nn.ModuleList([])
        for r in range(R):
            self.fusion_modules.append(
                FusionLayer(B, emb_dim, fusion_type)
            )
            for x in range(X):
                dilation = 2 ** x
                self.ctn_modules.append(
                    TemporalBlock(
                        input_shape=input_shape,
                        out_channels=H,
                        kernel_size=P,
                        stride=1,
                        padding="same",
                        dilation=dilation,
                        norm_type=norm_type,
                        causal=causal,
                        #  layer_name=f"temporalblock_{r}_{x}",
                    )
                )
                #  self.append(
                #      TemporalBlock,
                #      out_channels=H,
                #      kernel_size=P,
                #      stride=1,
                #      padding="same",
                #      dilation=dilation,
                #      norm_type=norm_type,
                #      causal=causal,
                #      layer_name=f"temporalblock_{r}_{x}",
                #  )
    def forward(self, inp, emb):
        '''
            inp: Tensor
                shape [M, K, B], K is the sequence length
            emb: Tensor
                shape [M, D]
        '''
        R = len(self.fusion_modules)
        X = len(self.ctn_modules) // R
        for r in range(R):
            # [M, K, B]
            inp = self.fusion_modules[r](inp, emb)
            for x in range(X):
                # [M, K, B]
                inp = self.ctn_modules[r*X+x](inp)
        return inp


class TemporalBlock(torch.nn.Module):
    """The conv1d compound layers used in Masknet.
    Arguments
    ---------
    input_shape : tuple
        The expected shape of the input.
    out_channels : int
        The number of intermediate channels.
    kernel_size : int
        The kernel size in the convolutions.
    stride : int
        Convolution stride in convolutional layers.
    padding : str
        The type of padding in the convolutional layers,
        (same, valid, causal). If "valid", no padding is performed.
    dilation : int
        Amount of dilation in convolutional layers.
    norm type : str
        The type of normalization, in ['gLN', 'cLN'].
    causal : bool
        To use causal or non-causal convolutions, in [True, False].
    Example:
    ---------
    >>> x = torch.randn(14, 100, 10)
    >>> TemporalBlock = TemporalBlock(x.shape, 10, 11, 1, 'same', 1)
    >>> y = TemporalBlock(x)
    >>> y.shape
    torch.Size([14, 100, 10])
    """

    def __init__(
        self,
        input_shape,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        norm_type="gLN",
        causal=False,
    ):
        super().__init__()
        M, K, B = input_shape

        #  self.layers = sb.nnet.containers.Sequential(input_shape=input_shape)
        #
        #  # [M, K, B] -> [M, K, H]
        #  self.layers.append(
        #      sb.nnet.CNN.Conv1d,
        #      out_channels=out_channels,
        #      kernel_size=1,
        #      bias=False,
        #      layer_name="conv",
        #  )
        #  self.layers.append(nn.PReLU(), layer_name="act")
        #  self.layers.append(
        #      choose_norm(norm_type, out_channels), layer_name="norm"
        #  )
        #
        #  # [M, K, H] -> [M, K, B]
        #  self.layers.append(
        #      DepthwiseSeparableConv,
        #      out_channels=B,
        #      kernel_size=kernel_size,
        #      stride=stride,
        #      padding=padding,
        #      dilation=dilation,
        #      norm_type=norm_type,
        #      causal=causal,
        #      layer_name="DSconv",
        #  )

        layers = []
        layers.append(
            sb.nnet.CNN.Conv1d(
                out_channels=out_channels,
                kernel_size=1,
                #  input_shape=input_shape,
                in_channels=B,
                bias=False,
            )
        )
        layers.append(nn.PReLU())
        layers.append(
            choose_norm(norm_type, out_channels)
        )
        layers.append(
            DepthwiseSeparableConv(
                in_channels=out_channels,
                out_channels=B,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                norm_type=norm_type,
                causal=causal,
            )
        )
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        """
        Arguments
        ---------
        x : Tensor
            Tensor shape is [M, K, B].
        Returns
        -------
        x : Tensor
            Tensor shape is [M, K, B].
        """
        residual = x
        x = self.layers(x)
        return x + residual


class DepthwiseSeparableConv(nn.Module):
    """Building block for the Temporal Blocks of Masknet in ConvTasNet.
    Arguments
    ---------
    input_shape : tuple
        Expected shape of the input.
    out_channels : int
        Number of output channels.
    kernel_size : int
        The kernel size in the convolutions.
    stride : int
        Convolution stride in convolutional layers.
    padding : str
        The type of padding in the convolutional layers,
        (same, valid, causal). If "valid", no padding is performed.
    dilation : int
        Amount of dilation in convolutional layers.
    norm type : str
        The type of normalization, in ['gLN', 'cLN'].
    causal : bool
        To use causal or non-causal convolutions, in [True, False].
    Example
    -------
    >>> x = torch.randn(14, 100, 10)
    >>> DSconv = DepthwiseSeparableConv(x.shape, 10, 11, 1, 'same', 1)
    >>> y = DSconv(x)
    >>> y.shape
    torch.Size([14, 100, 10])
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        norm_type="gLN",
        causal=False,
    ):
        super().__init__()


        #  # [M, K, H] -> [M, K, H]
        #  self.append(
        #      sb.nnet.CNN.Conv1d,
        #      out_channels=in_channels,
        #      kernel_size=kernel_size,
        #      stride=stride,
        #      padding=padding,
        #      dilation=dilation,
        #      groups=in_channels,
        #      bias=False,
        #      layer_name="conv_0",
        #  )
        #
        #  if causal:
        #      self.append(Chomp1d(padding), layer_name="chomp")
        #
        #  self.append(nn.PReLU(), layer_name="act")
        #  self.append(choose_norm(norm_type, in_channels), layer_name="act")
        #
        #  # [M, K, H] -> [M, K, B]
        #  self.append(
        #      sb.nnet.CNN.Conv1d,
        #      out_channels=out_channels,
        #      kernel_size=1,
        #      bias=False,
        #      layer_name="conv_1",
        #  )

        layers = []
        layers.append(
            sb.nnet.CNN.Conv1d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=in_channels,
                bias=False,
            )
        )
        if causal:
            layers.append(Chomp1d(padding))
        layers.append(nn.PReLU())
        layers.append(choose_norm(norm_type, in_channels))
        layers.append(
            sb.nnet.CNN.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                bias=False,
            )
        )
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class FusionLayer(nn.Module):
    def __init__(
        self,
        in_dim,
        emb_dim,
        fusion_type='cat'
    ):
        super().__init__()
        assert fusion_type in ['cat', 'add', 'mult']
        self.fusion_type = fusion_type
        if fusion_type == 'cat':
            self.layer = nn.Linear(in_dim + emb_dim, in_dim)
        elif fusion_type in ['add', 'mult']:
            self.layer = nn.Linear(emb_dim, in_dim)

    def forward(self, x, emb):
        '''
            args:
                x: Tensor with shape [B, K, N],
                    B is batch size, N is channel, K is chunk len
                emb: Tensor with shape [B, D]
                    D is embedder dimension
        '''
        # [B, K, D]
        emb = emb.unsqueeze(1).expand(-1, x.shape[1], -1)
        if self.fusion_type == 'cat':
            # [B, K, N+D]
            x = torch.cat([x, emb], dim=-1)
            # [B, K, N]
            out = self.layer(x)
        elif self.fusion_type == 'add':
            # [B, K, N]
            out = self.layer(emb)
            out = x + out
        elif self.fusion_type == 'mult':
            # [B, K, N]
            out = self.layer(emb)
            out = x * out
        return out

