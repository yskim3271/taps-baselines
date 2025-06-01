import math
from torch import nn
from torch.nn import functional as F
from models.demucs import rescale_module, upsample2, downsample2
from torchaudio.models.conformer import ConformerLayer

class seconformer(nn.Module):
    def __init__(self,
                 chin=1,
                 chout=1,
                 hidden=32,
                 depth=4,
                 conformer_dim=256,
                 conformer_ffn_dim=256,
                 conformer_num_attention_heads=4,
                 conformer_depth = 2,
                 depthwise_conv_kernel_size=31,
                 kernel_size=8,
                 stride=4,
                 resample=4,
                 growth=2,
                 dropout=0.1,
                 rescale=0.1,
                 normalize=True,
                 sample_rate=16_000):

        super().__init__()

        self.chin = chin
        self.chout = chout
        self.hidden = hidden
        self.depth = depth
        self.kernel_size = kernel_size
        self.stride = stride
        self.resample = resample
        self.growth = growth
        self.dropout = dropout
        self.conformer_dim = conformer_dim
        self.conformer_ffn_dim = conformer_ffn_dim
        self.conformer_num_attention_heads = conformer_num_attention_heads
        self.conformer_depth = conformer_depth
        self.depthwise_conv_kernel_size = depthwise_conv_kernel_size
        self.sample_rate = sample_rate
        self.normalize = normalize

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        ch_scale = 2 

        for index in range(depth):
            encode = []
            encode += [
                nn.Conv1d(chin, hidden, kernel_size, stride),
                nn.ReLU(),
                nn.Conv1d(hidden, hidden * ch_scale, 1), nn.GLU(1),
            ]
            self.encoder.append(nn.Sequential(*encode))

            decode = []
            decode += [
                nn.Conv1d(hidden, ch_scale * hidden, 1), nn.GLU(1), nn.ReLU(),
                nn.ConvTranspose1d(hidden, chout, kernel_size, stride),
            ]
            if index > 0:
                decode.append(nn.ReLU())
            self.decoder.insert(0, nn.Sequential(*decode))
            chout = hidden
            chin = hidden
            hidden = int(growth * hidden)

        self.conformers = nn.ModuleList()
        
        for index in range(conformer_depth):
            self.conformers.append(
                ConformerLayer(
                    input_dim=conformer_dim,
                    ffn_dim=conformer_ffn_dim,
                    num_attention_heads=conformer_num_attention_heads,
                    depthwise_conv_kernel_size=depthwise_conv_kernel_size,
                    dropout=dropout,
                )
            )
        if rescale:
            rescale_module(self, reference=rescale)

    def valid_length(self, length):
        """
        Return the nearest valid length to use with the model so that
        there is no time steps left over in a convolutions, e.g. for all
        layers, size of the input - kernel_size % stride = 0.

        If the mixture has a valid length, the estimated sources
        will have exactly the same length.
        """
        length = math.ceil(length * self.resample)
        for idx in range(self.depth):
            length = math.ceil((length - self.kernel_size) / self.stride) + 1
            length = max(length, 1)
        for idx in range(self.depth):
            length = (length - 1) * self.stride + self.kernel_size
        length = int(math.ceil(length / self.resample))
        return int(length)

    def forward(self, mix):
        if mix.dim() == 2:
            mix = mix.unsqueeze(1)
        if self.normalize:
            mono = mix.mean(dim=1, keepdim=True)
            std = mono.std(dim=-1, keepdim=True)
            mix = mix / (1e-3 + std)
        else:
            std = 1
        length = mix.shape[-1]
        x = mix
        x = F.pad(x, (0, self.valid_length(length) - length))

        if self.resample == 2:
            x = upsample2(x)
        elif self.resample == 4:
            x = upsample2(x)
            x = upsample2(x)
            
        skips = []
        for encode in self.encoder:
            x = encode(x)
            skips.append(x)
        x = x.permute(2, 0, 1)
        
        for conformer in self.conformers:
            x = conformer(x, None)
        
        x = x.permute(1, 2, 0)
        for decode in self.decoder:
            skip = skips.pop(-1)
            x = x + skip[..., :x.shape[-1]]
            x = decode(x)
        if self.resample == 2:
            x = downsample2(x)
        elif self.resample == 4:
            x = downsample2(x)
            x = downsample2(x)

        x = x[..., :length]
        return std * x
