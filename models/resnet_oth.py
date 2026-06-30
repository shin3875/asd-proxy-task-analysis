"""
Paper:      Deep Residual Learning for Image Recognition
Url:        https://arxiv.org/abs/1512.03385
Create by:  zh320
Date:       2024/07/13
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152


def _make_resnet(resnet_fn, pretrained: bool):
    try:
        return resnet_fn(weights="DEFAULT" if pretrained else None)
    except TypeError:
        return resnet_fn(pretrained=pretrained)


class ResNet(nn.Module):
    def __init__(self, num_class, resnet_type, num_channel=1, pretrained=False, downsample_rate=32):
        super(ResNet, self).__init__()
        resnet_hub = {'resnet18':resnet18, 'resnet34':resnet34, 'resnet50':resnet50,
                        'resnet101':resnet101, 'resnet152':resnet152}
        if resnet_type not in resnet_hub:
            raise ValueError(f'Unsupported ResNet type: {resnet_type}.\n')

        last_channel = 512 if resnet_type in ['resnet18', 'resnet34'] else 2048

        self.model = _make_resnet(resnet_hub[resnet_type], pretrained=pretrained)

        if num_channel != 3:
            self.model.conv1 = nn.Conv2d(in_channels=num_channel, 
                                        out_channels=self.model.conv1.out_channels, 
                                        kernel_size=self.model.conv1.kernel_size, 
                                        stride=self.model.conv1.stride, 
                                        padding=self.model.conv1.padding, 
                                        bias=self.model.conv1.bias)

        self.model.fc = nn.Linear(last_channel, num_class)

        if downsample_rate != 32:
            if downsample_rate in [8, 16]:
                self.model.conv1 = nn.Conv2d(num_channel, 64, 7, stride=1, padding=3, bias=False)
                if downsample_rate == 8:
                    self.model.maxpool = nn.Identity()
            else:
                raise NotImplementedError

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)

        features = torch.flatten(x, 1)

        logits = self.model.fc(features)

        return logits, features


class ResNetBack(nn.Module):
    def __init__(self, resnet_type, num_channel=1, pretrained=False, downsample_rate=32):
        super(ResNetBack, self).__init__()
        resnet_hub = {'resnet18':resnet18, 'resnet34':resnet34, 'resnet50':resnet50,
                        'resnet101':resnet101, 'resnet152':resnet152}
        if resnet_type not in resnet_hub:
            raise ValueError(f'Unsupported ResNet type: {resnet_type}.\n')

        self.model = _make_resnet(resnet_hub[resnet_type], pretrained=pretrained)

        if num_channel != 3:
            self.model.conv1 = nn.Conv2d(in_channels=num_channel,
                                        out_channels=self.model.conv1.out_channels,
                                        kernel_size=self.model.conv1.kernel_size,
                                        stride=self.model.conv1.stride,
                                        padding=self.model.conv1.padding,
                                        bias=self.model.conv1.bias)

        if downsample_rate != 32:
            if downsample_rate in [8, 16]:
                self.model.conv1 = nn.Conv2d(num_channel, 64, 7, stride=1, padding=3, bias=False)
                if downsample_rate == 8:
                    self.model.maxpool = nn.Identity()
            else:
                raise NotImplementedError

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)

        features = torch.flatten(x, 1)

        return features


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class SimCLR(nn.Module):
    def __init__(self, backbone, proj_hidden_dim=2048, proj_out_dim=128):
        super().__init__()

        self.projector = MLP(in_dim=proj_hidden_dim,
                             hidden_dim=proj_hidden_dim,
                             out_dim=proj_out_dim)
        self.backbone = backbone


    def forward(self, x):
        features = self.backbone(x)
        projections = self.projector(features)
        return projections


def nt_xent_loss(z_i, z_j, temperature):
    if z_i.shape != z_j.shape:
        raise ValueError(f"Shape mismatch: z_i={tuple(z_i.shape)}, z_j={tuple(z_j.shape)}")
    batch_size = z_i.shape[0]
    z = torch.cat([z_i, z_j], dim=0)
    z = F.normalize(z, p=2, dim=1)

    logits = torch.mm(z, z.T) / temperature
    logits.fill_diagonal_(float("-inf"))

    labels = torch.arange(2 * batch_size, device=z.device)
    labels = (labels + batch_size) % (2 * batch_size)
    return F.cross_entropy(logits, labels)


class SimSiam(nn.Module):
    def __init__(self, backbone, proj_hidden_dim=2048, proj_out_dim=2048, pred_hidden_dim=512):
        super().__init__()
        self.backbone = backbone

        self.projector = MLP(in_dim=proj_hidden_dim,
                             hidden_dim=proj_hidden_dim,
                             out_dim=proj_out_dim)
        self.predictor = MLP(in_dim=proj_out_dim,
                             hidden_dim=pred_hidden_dim,
                             out_dim=proj_out_dim)

        self.backbone.fc = nn.Identity()

    def forward(self, x1, x2):
        f = self.backbone
        z1 = self.projector(f(x1))
        z2 = self.projector(f(x2))

        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        return p1, z2.detach(), p2, z1.detach()


def simsiam_loss(p1, z2, p2, z1):
    loss1 = -F.cosine_similarity(p1, z2, dim=-1).mean()
    loss2 = -F.cosine_similarity(p2, z1, dim=-1).mean()
    return (loss1 + loss2) / 2
