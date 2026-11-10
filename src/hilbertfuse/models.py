"""Pretrained backbones adapted to 1-channel gene images.

All backbones come from torchvision with ImageNet weights (`pretrained=True`).
Each is wrapped so that
  * input is a (B,1,S,S) image in [0,1]; it is standardised and resized inside the model
    (so Guided Grad-CAM gradients are still w.r.t. the ORIGINAL pixels = genes),
  * all ReLUs are non-inplace modules (needed for guided back-propagation hooks),
  * `target_layer` points at the last post-ReLU feature map (Grad-CAM target).
"""
from __future__ import annotations
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm

SUPPORTED = ["hilbertfusenet", "hilbertfusenet121", "densenet_paper", "densenet121", "densenet169", "densenet201", "densenet161",
             "resnet18", "resnet34", "resnet50"]

_WEIGHTS = {  # torchvision weight enums (new API); falls back to string "DEFAULT"
    "densenet121": "DenseNet121_Weights", "densenet169": "DenseNet169_Weights",
    "densenet201": "DenseNet201_Weights", "densenet161": "DenseNet161_Weights",
    "resnet18": "ResNet18_Weights", "resnet34": "ResNet34_Weights", "resnet50": "ResNet50_Weights",
}


def _adapt_stem(conv: nn.Conv2d, in_channels: int) -> nn.Conv2d:
    return conv if in_channels == 3 else _collapse_conv_to_1ch(conv)


def _collapse_conv_to_1ch(conv: nn.Conv2d) -> nn.Conv2d:
    """Replace a 3-channel stem conv by a 1-channel one, summing the pretrained weights.
    (Keeps the learnt edge/texture filters; the summed filter responds to grey-scale input.)"""
    new = nn.Conv2d(1, conv.out_channels, conv.kernel_size, conv.stride, conv.padding, bias=conv.bias is not None)
    with torch.no_grad():
        new.weight.copy_(conv.weight.sum(dim=1, keepdim=True))
        if conv.bias is not None:
            new.bias.copy_(conv.bias)
    return new


def _no_inplace(module: nn.Module):
    for m in module.modules():
        if isinstance(m, nn.ReLU):
            m.inplace = False


class GeneCNN(nn.Module):
    def __init__(self, backbone: str, num_classes: int, pretrained: bool = True,
                 input_size: int = 96, dropout: float = 0.2, in_channels: int = 1):
        super().__init__()
        if backbone not in SUPPORTED:
            raise ValueError(f"backbone must be one of {SUPPORTED}")
        self.backbone_name, self.input_size = backbone, input_size
        net = self._load(backbone, pretrained)
        _no_inplace(net)
        self.relu = nn.ReLU(inplace=False)
        if backbone.startswith("densenet"):
            net.features.conv0 = _adapt_stem(net.features.conv0, in_channels)
            self.features = net.features
            n_feat = net.classifier.in_features
            self.family = "densenet"
            self.target_layer = self.relu           # post-ReLU final feature map
        else:
            net.conv1 = _adapt_stem(net.conv1, in_channels)
            self.features = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool,
                                          net.layer1, net.layer2, net.layer3, net.layer4)
            n_feat = net.fc.in_features
            self.family = "resnet"
            self.target_layer = net.layer4[-1]       # block ends with ReLU
        self.drop = nn.Dropout(dropout)
        self.classifier = nn.Linear(n_feat, num_classes)
        self.register_buffer("mean", torch.tensor(0.449))
        self.register_buffer("std", torch.tensor(0.226))

    @staticmethod
    def _load(name, pretrained):
        ctor = getattr(tvm, name)
        if not pretrained:
            return ctor(weights=None)
        try:
            enum = getattr(tvm, _WEIGHTS[name]).DEFAULT
            return ctor(weights=enum)
        except Exception as e:  # offline, blocked download, ...
            warnings.warn(f"Could not load pretrained weights for {name} ({e}). "
                          "Falling back to RANDOM initialisation. Pre-download the weights or "
                          "pass pretrained=False to silence this.")
            return ctor(weights=None)

    def forward(self, x):
        x = (x - self.mean) / self.std
        if x.shape[-1] != self.input_size:
            x = F.interpolate(x, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False)
        f = self.features(x)
        if self.family == "densenet":
            f = self.relu(f)
        f = F.adaptive_avg_pool2d(f, 1).flatten(1)
        return self.classifier(self.drop(f))


def build_model(backbone="hilbertfusenet", num_classes=2, pretrained=True, input_size=96, dropout=0.2,
                in_channels=1, **model_kwargs) -> nn.Module:
    """backbone: hilbertfusenet (compact, from scratch) | hilbertfusenet121 (pretrained DenseNet-121 + SE +
    multi-scale head) | densenet_paper (compact DenseNet without the new parts) | torchvision names."""
    from .hilbertfusenet import HilbertFuseNet
    if backbone == "hilbertfusenet":
        return HilbertFuseNet(num_classes, in_channels, "compact", **model_kwargs)
    if backbone == "densenet_paper":
        kw = dict(se=False, multiscale=False, drop_rate=0.0, head_dropout=0.0); kw.update(model_kwargs)
        return HilbertFuseNet(num_classes, in_channels, "compact", **kw)
    if backbone == "hilbertfusenet121":
        return HilbertFuseNet(num_classes, in_channels, "pretrained121", pretrained=pretrained,
                           input_size=input_size, **model_kwargs)
    return GeneCNN(backbone, num_classes, pretrained, input_size, dropout, in_channels)
