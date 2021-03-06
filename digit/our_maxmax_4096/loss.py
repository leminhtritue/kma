import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.functional as F
import pdb
import sys

def Entropy(input_):
    entropy = -input_ * torch.log(input_ + 1e-5)
    entropy = torch.sum(entropy, dim=1)
    return entropy 

def Entropy_1D(input_):
    entropy = -input_ * torch.log(input_ + 1e-5)
    entropy = torch.sum(entropy)
    return entropy 

class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True, size_average=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.size_average = size_average
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        if self.size_average:
            loss = (- targets * log_probs).mean(0).sum()
        else:
            loss = (- targets * log_probs).sum(1)
        return loss

class KernelSource(nn.Module):
    def __init__(self, num_classes, alpha = 0.1, use_gpu=True):
        super(KernelSource, self).__init__()
        self.num_classes = num_classes
        self.use_gpu = use_gpu
        self.alpha = alpha

    def forward(self, inputs, targets, hyperplanceNet):
        if self.use_gpu: 
            targets = targets.cuda()
        mark_multiply = torch.ones(inputs.size()).cuda()
        mark_add = torch.ones(inputs.size()).cuda()
        mark_cmp = torch.zeros(inputs.size()).cuda()
        for i in range(self.num_classes):
            mark_multiply[:, i][targets==i] = -1
            mark_add[:, i][targets==i] = 0

        loss_02 = torch.maximum(inputs * mark_multiply + mark_add, mark_cmp).mean(dim=0)
        loss_01 = 0.5* hyperplanceNet.get_weight().norm(dim=1)
        loss = self.alpha * loss_01.mean() + loss_02.mean()
        # print("loss 01 02: ", loss_01.mean().item(), loss_02.mean().item())
        return loss