"""HilbertFuseNet - a DenseNet redesigned for gene images.

Changes relative to the plain DenseNet used in the paper:
  1. multi-channel input (expression | MI prior | per-sample rank) with a learnable input norm;
  2. zero-initialised Squeeze-Excitation gates after every dense block (identity at start, so a
     pretrained backbone is not disturbed) -> the net can re-weight gene-module feature maps;
  3. multi-scale head: global-average features of block 2, block 3 and the last block are
     concatenated -> less sensitive to image size/layout (the instability seen in Table 6);
  4. compact from-scratch variant (small growth rate, stride-1 stem) suited to 60-1000 samples,
     and a pretrained variant built on torchvision DenseNet-121 ImageNet weights.
"""
from __future__ import annotations
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
from torchvision.models.densenet import _DenseBlock, _Transition


class SE(nn.Module):
    """Squeeze-Excitation gate, output = x * 2*sigmoid(fc(GAP(x))); identity at initialisation."""

    def __init__(self, c: int, reduction: int = 8):
        super().__init__()
        h = max(4, c // reduction)
        self.fc1, self.act, self.fc2 = nn.Linear(c, h), nn.ReLU(), nn.Linear(h, c)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        s = self.fc2(self.act(self.fc1(x.mean((2, 3)))))
        return x * (2.0 * torch.sigmoid(s))[:, :, None, None]


class HilbertFuseNet(nn.Module):
    def __init__(self, num_classes: int, in_channels: int = 3, variant: str = "compact",
                 pretrained: bool = False, input_size: int = 96,
                 growth_rate: int = 12, block_config=(4, 6, 8, 6), init_features: int = 24,
                 bn_size: int = 4, drop_rate: float = 0.1, se: bool = True, multiscale: bool = True,
                 head_dropout: float = 0.3, se_reduction: int = 8, probe_size: int = 32):
        super().__init__()
        assert variant in ("compact", "pretrained121")
        self.variant, self.input_size, self.multiscale = variant, input_size, multiscale
        self.family = "hilbertfusenet"
        if variant == "compact":
            self.resize = None
            self.input_norm = nn.BatchNorm2d(in_channels)
            self.stem = nn.Sequential(nn.Conv2d(in_channels, init_features, 3, 1, 1, bias=False),
                                      nn.BatchNorm2d(init_features), nn.ReLU())
            nf, blocks, trans = init_features, [], []
            for i, nl in enumerate(block_config):
                blocks.append(_DenseBlock(nl, nf, bn_size, growth_rate, drop_rate))
                nf += nl * growth_rate
                if i < len(block_config) - 1:
                    trans.append(_Transition(nf, nf // 2))
                    nf //= 2
            self.final_norm = nn.BatchNorm2d(nf)
        else:
            self.resize = input_size
            self.register_buffer("mean", torch.tensor(0.449))
            self.register_buffer("std", torch.tensor(0.226))
            net = self._load_tv(pretrained)
            f = net.features
            conv0 = f.conv0
            if in_channels != 3:
                new = nn.Conv2d(in_channels, conv0.out_channels, conv0.kernel_size, conv0.stride,
                                conv0.padding, bias=False)
                with torch.no_grad():
                    w = conv0.weight.sum(1, keepdim=True)
                    new.weight.copy_(w.repeat(1, in_channels, 1, 1) / in_channels)
                conv0 = new
            self.stem = nn.Sequential(conv0, f.norm0, f.relu0, f.pool0)
            blocks = [f.denseblock1, f.denseblock2, f.denseblock3, f.denseblock4]
            trans = [f.transition1, f.transition2, f.transition3]
            self.final_norm = f.norm5
        self.blocks, self.trans = nn.ModuleList(blocks), nn.ModuleList(trans)
        for m in self.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = False
        self.relu = nn.ReLU()
        self.target_layer = self.relu

        # probe channel counts of every stage with a dummy pass
        with torch.no_grad():
            was = self.training
            self.eval()
            dims = self._probe(torch.zeros(2, in_channels, probe_size if variant == "compact" else input_size,
                                           probe_size if variant == "compact" else input_size))
            self.train(was)
        self.ses = nn.ModuleList([SE(c, se_reduction) if se else nn.Identity() for c in dims["blocks"]])
        self.tap_ids = [1, 2] if (multiscale and len(blocks) >= 4) else []
        self.tap_bn = nn.ModuleList([nn.BatchNorm2d(dims["blocks"][i]) for i in self.tap_ids])
        self.tap_act = nn.ModuleList([nn.ReLU() for _ in self.tap_ids])
        feat = dims["final"] + sum(dims["blocks"][i] for i in self.tap_ids)
        self.drop = nn.Dropout(head_dropout)
        self.classifier = nn.Linear(feat, num_classes)

    @staticmethod
    def _load_tv(pretrained):
        if not pretrained:
            return tvm.densenet121(weights=None)
        try:
            return tvm.densenet121(weights=tvm.DenseNet121_Weights.DEFAULT)
        except Exception as e:
            warnings.warn(f"Could not load pretrained DenseNet-121 weights ({e}); using RANDOM init.")
            return tvm.densenet121(weights=None)

    def _prep(self, x):
        if self.variant == "compact":
            return self.input_norm(x)
        x = (x - self.mean) / self.std
        if x.shape[-1] != self.resize:
            x = F.interpolate(x, size=(self.resize, self.resize), mode="bilinear", align_corners=False)
        return x

    def _probe(self, x):
        x = self.stem(self._prep(x))
        d = []
        for i, b in enumerate(self.blocks):
            x = b(x)
            d.append(x.shape[1])
            if i < len(self.trans):
                x = self.trans[i](x)
        return {"blocks": d, "final": self.final_norm(x).shape[1]}

    def forward(self, x):
        x = self.stem(self._prep(x))
        taps = []
        for i, b in enumerate(self.blocks):
            x = self.ses[i](b(x))
            taps.append(x)
            if i < len(self.trans):
                x = self.trans[i](x)
        f = F.adaptive_avg_pool2d(self.relu(self.final_norm(x)), 1).flatten(1)
        feats = [f]
        for k, i in enumerate(self.tap_ids):
            feats.append(F.adaptive_avg_pool2d(self.tap_act[k](self.tap_bn[k](taps[i])), 1).flatten(1))
        return self.classifier(self.drop(torch.cat(feats, 1)))
