from torch import nn

from . import misc
from . import torch_util as tu


class MLP(nn.Module):
    def __init__(self, insize, nhidlayer, outsize, hidsize, hidactiv):
        super().__init__()
        self.insize = insize
        self.nhidlayer = nhidlayer
        self.outsize = outsize
        in_sizes = [insize] + [hidsize] * nhidlayer
        out_sizes = [hidsize] * nhidlayer + [outsize]
        self.layers = nn.ModuleList(
            [
                tu.NormedLinear(insize, outsize)
                for (insize, outsize) in misc.safezip(in_sizes, out_sizes)
            ]
        )
        self.hidactiv = hidactiv

    def forward(self, x):
        *hidlayers, finallayer = self.layers
        for layer in hidlayers:
            x = layer(x)
            x = self.hidactiv(x)
        x = finallayer(x)
        return x

    @property
    def output_shape(self):
        return (self.outsize,)
