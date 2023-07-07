import numpy as np

import torch
import torch.nn as nn

from torchvision import transforms, utils, models


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


class Vgg19(nn.Module):
    conv_layers = {"conv1": 6, "conv2": 13, "conv3": 26, "conv4": 39, "conv5": 52}
    fc_layers = {"fc6": 1, "fc7": 4}

    def __init__(self, layer, extract_conv=True):
        super(Vgg19, self).__init__()
        self.extract_conv = extract_conv
        if self.extract_conv:
            self.layer_ind = Vgg19.conv_layers[layer]
        else:
            self.layer_ind = Vgg19.fc_layers[layer]

        # load models from PyTorch
        vgg19_bn = models.vgg19_bn(pretrained=True)
        vgg19_bn.to(device)
        for param in vgg19_bn.parameters():
            param.requires_grad = False
        vgg19_bn.eval()

        features = list(vgg19_bn.features)
        self.features = nn.ModuleList(features).eval()
        self.adaptivepool = vgg19_bn.avgpool
        if (
            not self.extract_conv
        ):  # if need fc layer then add those linear layers into the forward pass
            self.classifiers = nn.Sequential(
                *list(vgg19_bn.classifier.children())
            ).eval()

    def forward(self, x, subsample, subsampling_size=20000):
        results = []
        for ii, layer in enumerate(self.features):
            x = layer(x)
            if self.extract_conv and self.layer_ind == ii:
                if subsample == "avgpool":
                    # print(x.data.shape)
                    c = x.data.shape[1]  # number of channels
                    k = int(np.floor(np.sqrt(subsampling_size / c)))
                    results = (
                        nn.functional.adaptive_avg_pool2d(x.data, (k, k))
                        .cpu()
                        .flatten()
                        .numpy()
                    )
                elif subsample == "pca":
                    if (
                        self.layer_ind == Vgg19.conv_layers["conv1"]
                    ):  # need to reduce dimension of the first layer by half for PCA
                        results = (
                            nn.functional.avg_pool2d(x.data, (2, 2))
                            .cpu()
                            .flatten()
                            .numpy()
                            .astype(np.float16)
                        )
                    else:
                        results = x.cpu().flatten().numpy().astype(np.float16)
                else:
                    results = x.cpu().flatten().numpy()
                break

        if not self.extract_conv:
            x = self.adaptivepool(x)
            x = x.view(-1)
            for ii, layer in enumerate(self.classifiers):
                x = layer(x)
                if self.layer_ind == ii:
                    results = x.view(-1).data.cpu().numpy()
                    break
        return results


class AlexNet(nn.Module):
    conv_layers = {"conv1": 2, "conv2": 5, "conv3": 7, "conv4": 9, "conv5": 12}
    fc_layers = {"fc6": 1, "fc7": 4}

    def __init__(self, layer, extract_conv=True):
        super(AlexNet, self).__init__()
        self.extract_conv = extract_conv
        if self.extract_conv:
            self.layer_ind = AlexNet.conv_layers[layer]
        else:
            self.layer_ind = AlexNet.fc_layers[layer]

        # load models from PyTorch
        alexnet = models.alexnet(pretrained=True)
        alexnet.to(device)
        for param in alexnet.parameters():
            param.requires_grad = False
        alexnet.eval()

        features = list(alexnet.features)
        self.features = nn.ModuleList(features).eval()
        self.adaptivepool = alexnet.avgpool
        if (
            not self.extract_conv
        ):  # if need fc layer then add those linear layers into the forward pass
            self.classifiers = nn.Sequential(
                *list(alexnet.classifier.children())
            ).eval()

    def forward(self, x, subsample, subsampling_size=20000):
        results = []
        for ii, layer in enumerate(self.features):
            x = layer(x)
            if self.extract_conv and self.layer_ind == ii:
                if subsample == "avgpool":
                    # print(x.data.shape)
                    c = x.data.shape[1]  # number of channels
                    k = int(np.floor(np.sqrt(subsampling_size / c)))
                    results = (
                        nn.functional.adaptive_avg_pool2d(x.data, (k, k))
                        .cpu()
                        .flatten()
                        .numpy()
                    )
                elif subsample == "pca":
                    if (
                        self.layer_ind == AlexNet.conv_layers["conv1"]
                    ):  # need to reduce dimension of the first layer by half for PCA
                        results = (
                            nn.functional.avg_pool2d(x.data, (2, 2))
                            .cpu()
                            .flatten()
                            .numpy()
                            .astype(np.float16)
                        )
                    else:
                        results = x.cpu().flatten().numpy().astype(np.float16)
                else:
                    results = x.cpu().flatten().numpy()
                break

        if not self.extract_conv:
            x = self.adaptivepool(x)
            x = x.view(-1)
            for ii, layer in enumerate(self.classifiers):
                x = layer(x)
                if self.layer_ind == ii:
                    results = x.view(-1).data.cpu().numpy()
                    break
        return results
