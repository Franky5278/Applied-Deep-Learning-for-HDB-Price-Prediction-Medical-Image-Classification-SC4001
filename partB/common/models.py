import torch
from torch import nn


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=3, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel, stride=stride, padding=kernel // 2, bias=False)  # 卷积不使用偏置。
        self.bn = nn.BatchNorm2d(out_channels)  # Normalize each output channel.
        self.relu = nn.ReLU()

    def forward(self, images):
        return self.relu(self.bn(self.conv(images)))


class AlexNetLike(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = ConvBNReLU(1, 16, 5)
        self.conv2 = ConvBNReLU(16, 32, 5)
        self.conv3 = ConvBNReLU(32, 64, 3)
        self.pool = nn.MaxPool2d(2)
        self.average = nn.AdaptiveAvgPool2d((4, 4))  # Pool to the required 4 x 4 grid.
        self.classifier = nn.Sequential(
            # Dropout 仅用于两个隐藏全连接层。
            nn.Flatten(), nn.Linear(64 * 4 * 4, 64), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.5), nn.Linear(32, 1),
        )

    def forward(self, images):
        features = self.pool(self.conv1(images))
        features = self.pool(self.conv2(features))
        features = self.pool(self.conv3(features))
        return self.classifier(self.average(features)).squeeze(1)


class VGGLike(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # 每组两层卷积后做池化。
            ConvBNReLU(1, 8), ConvBNReLU(8, 8), nn.MaxPool2d(2),
            ConvBNReLU(8, 16), ConvBNReLU(16, 16), nn.MaxPool2d(2),
            ConvBNReLU(16, 32), ConvBNReLU(32, 32), nn.MaxPool2d(2),
        )
        self.average = nn.AdaptiveAvgPool2d(1)  # Global average pooling.
        self.fc = nn.Linear(32, 1)

    def forward(self, images):
        return self.fc(self.average(self.features(images)).flatten(1)).squeeze(1)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = ConvBNReLU(in_channels, out_channels, stride=stride)
        self.conv2 = ConvBNReLU(out_channels, out_channels)
        self.shortcut = ConvBNReLU(in_channels, out_channels, kernel=1, stride=stride) if stride != 1 or in_channels != out_channels else nn.Identity()  # 投影支路同步改变通道和尺寸。
        self.relu = nn.ReLU()

    def forward(self, features):
        return self.relu(self.conv2(self.conv1(features)) + self.shortcut(features))  # Activate after residual addition.


class ResNetLike(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = ConvBNReLU(1, 16)
        self.pool = nn.MaxPool2d(2)
        self.block1 = ResidualBlock(16, 16)
        self.block2 = ResidualBlock(16, 32, stride=2)
        self.block3 = ResidualBlock(32, 64, stride=2)
        self.average = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, 1)

    def forward(self, images):
        features = self.pool(self.stem(images))
        features = self.block3(self.block2(self.block1(features)))
        return self.fc(self.average(features).flatten(1)).squeeze(1)


MODELS = {"b1": AlexNetLike, "b2": VGGLike, "b3": ResNetLike}
DEFAULT_TARGETS = {"b1": "conv3.conv", "b2": "features.7.conv", "b3": "block3.conv2.conv"}


def complexity(model):
    # 统一统计卷积和全连接的计算量。
    rows = []
    hooks = []
    for name, module in model.named_modules():
        if not isinstance(module, (nn.Conv2d, nn.Linear)):
            continue
        def hook(layer, inputs, output, layer_name=name):
            if isinstance(layer, nn.Conv2d):
                macs = output.numel() * (layer.in_channels // layer.groups) * layer.kernel_size[0] * layer.kernel_size[1]
            else:
                macs = output.numel() * layer.in_features
            rows.append({"layer": layer_name, "macs": int(macs), "flops": int(2 * macs)})  # Count each multiply-add as two FLOPs.
        hooks.append(module.register_forward_hook(hook))
    previous = model.training
    model.eval()
    device = next(model.parameters()).device
    try:
        with torch.no_grad():
            model(torch.zeros(1, 1, 64, 64, device=device))
    finally:
        for handle in hooks:
            handle.remove()
        model.train(previous)
    return {
        "learnable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "macs": sum(row["macs"] for row in rows),
        "flops": sum(row["flops"] for row in rows),
        "convention": "单张 1×64×64 输入；仅 Conv2d/Linear；一次乘加计 2 FLOPs；不计 bias、BN、ReLU、pool、残差相加及 sigmoid",
        "layers": rows,
    }
