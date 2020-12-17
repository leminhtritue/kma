import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.functional as F
import pdb
import sys

def Entropy(input_):
    bs = input_.size(0)
    entropy = -input_ * torch.log(input_ + 1e-5)
    entropy = torch.sum(entropy, dim=1)
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
        loss = 0.0
        for i in range(self.num_classes):
            mark_multiply = torch.ones(targets.size()).cuda()
            mark_multiply[targets==i] = -1
            mark_add = torch.zeros(targets.size()).cuda()
            mark_multiply[targets==i] = 1
            temp_value = inputs[:,i] * mark_multiply + mark_add
            mark_cmp = torch.zeros(temp_value.size()).cuda()
            loss += torch.minimum(temp_value, mark_cmp).sum()
        print(loss)

        tt = hyperplanceNet.get_weight().norm(dim=1)
        print(tt.shape)
        print(type(tt))
        print(tt)
        sys.exit()
        return loss