import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.functional as F
import pdb

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
        if self.use_gpu: targets = targets.cuda()
        loss = 0.0
        print(targets.shape)
        print(targets.min)
        print(targets.max)
        for i in range(self.num_classes):
            mark_multiply = torch.ones(targets.size())
            mark_multiply[targets==i] = -1
            print(mark_multiply)
            print(mark_multiply.shape)
            sys.exit()
        return loss