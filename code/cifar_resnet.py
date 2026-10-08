"""
CIFAR ResNet-20 / ResNet-56 (He et al. 2016, CIFAR variant: 3 stages of 16/32/64
channels, stride-2 downsampling between stages, no initial 7x7/maxpool stem).
Architecture and state_dict key names matched to the public pretrained checkpoints
at https://github.com/chenyaofo/pytorch-cifar-models (cifar10_resnet20, cifar10_resnet56)
so the released weights load with strict=True -- this is the check that proves the
architecture is right, not an assumption.
"""
import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or in_c != out_c:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride, bias=False), nn.BatchNorm2d(out_c))

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class CifarResNet(nn.Module):
    def __init__(self, layers, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(16, 16, layers[0], 1)
        self.layer2 = self._make_layer(16, 32, layers[1], 2)
        self.layer3 = self._make_layer(32, 64, layers[2], 2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, in_c, out_c, n, stride):
        blocks = [BasicBlock(in_c, out_c, stride)]
        for _ in range(1, n):
            blocks.append(BasicBlock(out_c, out_c, 1))
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x)
        x = self.avgpool(x).flatten(1)
        return self.fc(x)


def resnet20(num_classes=10):
    return CifarResNet([3, 3, 3], num_classes)


def resnet56(num_classes=10):
    return CifarResNet([9, 9, 9], num_classes)


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def load_pretrained(arch, ckpt_path, device="cpu"):
    model = {"resnet20": resnet20, "resnet56": resnet56}[arch]()
    sd = torch.load(ckpt_path, map_location=device)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    model.eval()
    return model, missing, unexpected
