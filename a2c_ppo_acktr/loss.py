# ------------------------------------------------------------------------------
# Copyright (c) by contributors 
# Licensed under the MIT License.
# Written by Haiping Wu
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn


class SaliencyMapMSELoss(nn.Module):
    def __init__(self, use_target_weight, loss_weight=1.0):
        super(SaliencyMapMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='none')
        self.use_target_weight = use_target_weight
        self.loss_weight = loss_weight

        self.sum_cri = nn.MSELoss(reduction='sum')

    def forward(self, output, target, target_weight=None, threshold=None, target_mul_weight=True):
        batch_size = output.size(0)
        c = output.size(1)
        if self.use_target_weight:
            c_weight = target_weight.shape[1]
        if len(output.shape) == 5:
            batch_size = batch_size * c
            c = output.shape[2]
            if self.use_target_weight:
                c_weight = target_weight.shape[2]
        output = output.reshape((batch_size, c, -1))
        target = target.reshape((batch_size, c, -1))
        if self.use_target_weight:
            target_weight = target_weight.reshape((batch_size, c_weight, -1))

        if self.use_target_weight:
            if target.shape != target_weight.shape:
                target_weight = target_weight.repeat(1, c, 1)
            target_weight = target_weight.reshape((batch_size, c, -1))
            if not target_mul_weight:
                loss = self.criterion(output * target_weight,
                                      target)
            else:
                loss = self.criterion(output * target_weight,
                                      target * target_weight)
            if threshold is not None:
                loss = (loss > threshold) * loss
            loss = loss.mean()
        else:
            output = output.reshape((batch_size, -1))
            target = target.reshape((batch_size, -1))
            loss = self.criterion(output,
                                  target)
            if threshold is not None:
                loss = (loss > float(threshold)) * loss
            loss = loss.mean()
        return loss * self.loss_weight





