import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torch.autograd import Variable
import math
import torch.nn.utils.weight_norm as weightNorm
from collections import OrderedDict

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

class RandomFourierFeatures(nn.Module):
    def __init__(self, query_dimensions, n_dims=None, gamma = 0.5,
                 orthogonal=False):
        super(RandomFourierFeatures, self).__init__()

        self.n_dims = n_dims
        self.orthogonal = orthogonal
        self.gamma = gamma

        # Make a buffer for storing the sampled omega
        self.register_buffer(
            "omega",
            torch.zeros(query_dimensions, self.n_dims//2)
        )

    def new_feature_map(self):
        if self.orthogonal:
            orthogonal_random_matrix_(self.omega)
        else:
            self.omega.normal_()

    def forward(self, x):
        x = x * self.gamma
        u = x.matmul(self.omega)
        phi = torch.cat([torch.cos(u), torch.sin(u)], dim=-1)
        return phi

class feat_bootleneck(nn.Module):
    def __init__(self, feature_dim, bottleneck_dim=256, type="ori"):
        super(feat_bootleneck, self).__init__()
        self.bn = nn.BatchNorm1d(bottleneck_dim, affine=True)
        self.dropout = nn.Dropout(p=0.5)
        self.bottleneck = nn.Linear(feature_dim, bottleneck_dim)
        self.bottleneck.apply(init_weights)
        self.type = type

    def forward(self, x):
        x = self.bottleneck(x)
        if self.type == "bn":
            x = self.bn(x)
            x = self.dropout(x)
        return x

class feat_bootleneck_rf(nn.Module):
    def __init__(self, gamma = 0.5, bottleneck_dim=256, type="ori", nrf = 512):
        super(feat_bootleneck_rf, self).__init__()
        self.bn2 = nn.BatchNorm1d(nrf, affine=True)
        self.dropout2 = nn.Dropout(p=0.5)
        self.type = type

        self.feature_map = RandomFourierFeatures(query_dimensions = bottleneck_dim, n_dims = nrf, gamma = gamma)
        self.feature_map.new_feature_map()

    def forward(self, x):
        x = self.feature_map(x)
        if self.type == "bn":
            x = self.bn2(x)
            x = self.dropout2(x)
        return x

class feat_classifier(nn.Module):
    def __init__(self, class_num, bottleneck_dim=256, type="linear"):
        super(feat_classifier, self).__init__()
        if type == "linear":
            self.fc = nn.Linear(bottleneck_dim, class_num)
        else:
            self.fc = weightNorm(nn.Linear(bottleneck_dim, class_num), name="weight")
        self.fc.apply(init_weights)

    def forward(self, x):
        x = self.fc(x)
        return x

class feat_classifier_rf(nn.Module):
    def __init__(self, class_num, nrf = 512, type="linear"):
        super(feat_classifier_rf, self).__init__()
        self.type = type
        if type == 'wn':
            self.fc = weightNorm(nn.Linear(nrf, class_num), name="weight")
            self.fc.apply(init_weights)
        else:
            self.fc = nn.Linear(nrf, class_num)
            self.fc.apply(init_weights)

    def forward(self, x):
        x = self.fc(x)
        return x

    def get_weight(self):
        return self.fc.weight

    def get_bias(self):
        return self.fc.bias
        
class DTNBase(nn.Module):
    def __init__(self):
        super(DTNBase, self).__init__()
        self.conv_params = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm2d(64),
                nn.Dropout2d(0.1),
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm2d(128),
                nn.Dropout2d(0.3),
                nn.ReLU(),
                nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm2d(256),
                nn.Dropout2d(0.5),
                nn.ReLU()
                )   
        self.in_features = 256*4*4

    def forward(self, x):
        x = self.conv_params(x)
        x = x.view(x.size(0), -1)
        return x

class LeNetBase(nn.Module):
    def __init__(self):
        super(LeNetBase, self).__init__()
        self.conv_params = nn.Sequential(
                nn.Conv2d(1, 20, kernel_size=5),
                nn.MaxPool2d(2),
                nn.ReLU(),
                nn.Conv2d(20, 50, kernel_size=5),
                nn.Dropout2d(p=0.5),
                nn.MaxPool2d(2),
                nn.ReLU(),
                )
        self.in_features = 50*4*4

    def forward(self, x):
        x = self.conv_params(x)
        x = x.view(x.size(0), -1)
        return x