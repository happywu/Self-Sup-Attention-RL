# ------------------------------------------------------------------------------
# Copyright (c) by contributors 
# Licensed under the MIT License.
# Written by Haiping Wu
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import PIL
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
try:
    from matplotlib import animation, cm
except:
    print('cannot import matplotlib animation')

logger = logging.getLogger(__name__)

def calc_auc(preds, labels):
    fpr, tpr, thresholds = metrics.roc_curve(labels, preds)
    return metrics.auc(fpr, tpr)

def plot_keypoints_on_image(im, kpts, r=5, alpha=1.0, image_size=(128, 128)):
    im = (im.copy() * alpha).astype(np.uint8)
    scaled_kpts = kpts.copy()
    scaled_kpts = (scaled_kpts + 1.0) * 0.5 * np.array(list(image_size))[None, :]
    scaled_kpts = scaled_kpts.astype(np.int)
    temp = scaled_kpts.copy()
    if not isinstance(im, Image.Image):
        im = Image.fromarray(im)
    im = im.convert('RGB')

    im = im.resize(image_size, PIL.Image.BILINEAR)
    n_kpts = scaled_kpts.shape[0]
    colors = [tuple(c) for c in (cm.rainbow(np.linspace(0, 1, n_kpts)) * 255).astype(np.int32)]
    draw = ImageDraw.Draw(im)
    for i in range(len(scaled_kpts)):
        y, x = scaled_kpts[i].astype(np.int32)
        draw.ellipse((x - r, y - r, x + r, y + r), outline='red')
    return np.array(im)

# markdown format output
def _print_name_value(name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    logger.info('|---' * (num_values + 1) + '|')

    if len(full_arch_name) > 15:
        full_arch_name = full_arch_name[:8] + '...'
    logger.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
        ' |'
    )


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
