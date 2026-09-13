import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """
    Basic residual block cho ResNet tự xây dựng.

    Nhánh chính: Conv -> BatchNorm -> ReLU -> Conv -> BatchNorm.
    Nhánh shortcut: identity nếu shape khớp, hoặc Conv 1x1 nếu cần đổi kênh/stride.
    """

    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels * self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels * self.expansion),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        shortcut = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Residual connection bắt buộc của BasicBlock.
        out = F.relu(out + shortcut)
        return out


class CustomResNet(nn.Module):
    """ResNet-18 mini tự code từ đầu, phù hợp ảnh khuôn mặt 64x64."""

    def __init__(self, block, layers, num_classes=8, in_channels=3, dropout=0.3):
        super().__init__()

        self.in_channels = 64

        # Stem nhỏ hơn ResNet chuẩn để hợp với ảnh nhỏ 64x64.
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """Tạo một stage gồm nhiều BasicBlock liên tiếp."""
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []

        for block_stride in strides:
            blocks.append(block(self.in_channels, out_channels, block_stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


def custom_resnet18(num_classes, in_channels=3):
    """ResNet-18 tự xây dựng từ BasicBlock với số block [2, 2, 2, 2]."""
    return CustomResNet(
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        num_classes=num_classes,
        in_channels=in_channels,
        dropout=0.3,
    )


def hsemotion_efficientnet_b0(num_classes):
    import timm

    return timm.create_model("efficientnet_b0", pretrained=False, num_classes=num_classes)


def build_model(num_classes, architecture="custom_resnet"):
    """Dựng đúng kiến trúc theo metadata lưu trong checkpoint."""
    if architecture == "hsemotion_enet_b0_8_best_vgaf":
        return hsemotion_efficientnet_b0(num_classes=num_classes)
    if architecture == "efficientnet_b0":
        from torchvision.models import efficientnet_b0

        model = efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model
    if architecture in {"convnext_tiny", "efficientnet_b2", "resnet50"}:
        import timm

        return timm.create_model(architecture, pretrained=False, num_classes=num_classes)
    if architecture in {"custom_resnet", "custom_resnet18", None}:
        return custom_resnet18(num_classes=num_classes, in_channels=3)
    raise ValueError(f"Kiến trúc checkpoint chưa được hỗ trợ: {architecture}")

