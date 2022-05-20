from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import yaml

import numpy as np
from easydict import EasyDict as edict


config = edict()

config.AUTO_RESUME = False
config.OUTPUT_DIR = ''
config.LOG_DIR = ''
config.DATA_DIR = ''
config.GPUS = '0'
config.WORKERS = 4
config.PRINT_FREQ = 20
config.PIN_MEMORY = True

config.cuda = True
config.cuda_deterministic = False

config.algo = 'a2c'
config.gail = False
config.gail_experts_dir = './gail_experts'
config.gail_batch_size = 128
config.gail_epoch = 5
config.lr = 0.0007
config.eps = 0.00001
config.alpha = 0.99
config.weight_decay = 0.0
config.gamma = 0.99
config.use_gae = False
config.gae_lambda = 0.95
config.entropy_coef = 0.01
config.value_loss_coef = 0.5
config.max_grad_norm = 0.5
config.seed = 1
config.cuda_deterministic = False
config.num_processes = 16
config.num_steps = 5
config.ppo_epoch = 4
config.log_interval=10
config.save_interval=100
config.eval_interval=1245
config.num_env_steps = 10000000
config.env_name = 'PongNoFrameskip-v4'
config.save_dir = './trained_models/'
config.no_cuda = False
config.use_proper_time_limits = False
config.recurrent_policy = False
config.use_linear_lr_decay = False
config.hidden_size = 512

config.feat_pool_with_selfsup_attention = False
config.feat_from_selfsup_attention = False
config.feat_add_selfsup_attention = False
config.feat_mul_selfsup_attention_mask = False
config.selfsup_attention_fix = True
config.selfsup_attention_pretrain = ''
config.selfsup_attention_fix_keypointer = False
config.selfsup_attention_keyp_maps_pool = False
config.selfsup_attention_image_feat_only = False
config.selfsup_attention_feat_masked = False
config.selfsup_attention_feat_masked_residual = False
config.selfsup_attention_multiframe = False
config.selfsup_attention_feat_load_pretrained = True
config.use_layer_norm = False
config.feat_mul_selfsup_attention_mask_residual = True
config.bottom_up_form_objects = False
config.bottom_up_form_num_of_objects = 10
config.gaussian_std = 0.1
config.bottom_up_downsample_mask = False
config.bottom_up_downsample_mask_stride1 = False

config.train_selfsup_attention = False
config.block_selfsup_attention_grad = True
config.sep_bg_fg_feat = False
config.mask_threshold = -1.
config.fix_feature = False

config.train_selfsup_attention_buffer_size = 60

config.RESUME = False
config.MODEL_FILE = ''


# Cudnn related params
config.CUDNN = edict()
config.CUDNN.BENCHMARK = True
config.CUDNN.DETERMINISTIC = False
config.CUDNN.ENABLED = True

config.MODEL = edict()
config.MODEL.NAME = 'a2c'
config.MODEL.INIT_WEIGHTS = True
config.MODEL.PRETRAINED = ''
config.MODEL.IMAGE_SIZE = [256, 256]  # width * height, ex: 192 * 256
config.MODEL.HEATMAP_SIZE = [64, 64]  # width * height, ex: 24 * 32
config.MODEL.IGNORE_BACKGROUND = False
config.MODEL.SIGMA = 1

config.DATASET = edict()
config.DATASET.ROOT = ''
config.DATASET.DATASET = 'mpii'
config.DATASET.TRAIN_SET = 'train'
config.DATASET.TEST_SET = 'valid'
config.DATASET.SAMPLE_STEP = 1
config.DATASET.HISTORY_INPUT = False
config.DATASET.SAMPLE_2_FRAME = False
config.DATASET.SAMPLE_2_FRAME_OFFSET = 0
config.DATASET.GRAYSCALE = True
config.DATASET.HYBRID = False


# training data augmentation
config.DATASET.COLOR_RGB = True
config.DATASET.BRIGHTNESS_FACTOR = 0.
config.DATASET.CONTRAST_FACTOR = 0.
config.DATASET.HUE_FACTOR = 0.
config.DATASET.SAT_FACTOR = 0.
config.DATASET.TRANSFORM = False

config.SELFSUP_ATTENTION = edict()
config.SELFSUP_ATTENTION.NUM_KEYPOINTS = 10
config.SELFSUP_ATTENTION.GAUSS_STD = 0.11
config.SELFSUP_ATTENTION.AUTOENCODER_LOSS = False
config.SELFSUP_ATTENTION.KEYPOINTER_CLS_AGNOSTIC = False
config.SELFSUP_ATTENTION.HEATMAP_SPARSITY_LOSS = False
config.SELFSUP_ATTENTION.HEATMAP_SPARSITY_LOSS_WEIGHT = 0.
config.SELFSUP_ATTENTION.USE_LAYER_NORM = True
config.SELFSUP_ATTENTION.USE_INSTANCE_NORM = False
config.SELFSUP_ATTENTION.BLOCK_IMAGE_FEAT_GRADIENT = False
config.SELFSUP_ATTENTION.RECONSTRUCT_LOSS_THRESHOLD = 0.01
config.SELFSUP_ATTENTION.DECODER_WITH_SIGMOID = True
config.SELFSUP_ATTENTION.SHARE_PARAMETERS = False

config.TRAIN = edict()

config.TRAIN.LR_FACTOR = 0.1
config.TRAIN.LR_STEP = [90, 110]
config.TRAIN.LR = 0.001

config.TRAIN.OPTIMIZER = 'adam'
config.TRAIN.MOMENTUM = 0.9
config.TRAIN.WD = 0.0001
config.TRAIN.NESTEROV = False
config.TRAIN.GAMMA1 = 0.99
config.TRAIN.GAMMA2 = 0.0

config.TRAIN.BEGIN_EPOCH = 0
config.TRAIN.END_EPOCH = 140

config.TRAIN.RESUME = False
config.TRAIN.CHECKPOINT = ''

config.TRAIN.BATCH_SIZE_PER_GPU = 32
config.TRAIN.SHUFFLE = True

# testing
config.TEST = edict()

# size of images for each device
config.TEST.BATCH_SIZE_PER_GPU = 32
config.TEST.MODEL_FILE = ''

config.DEBUG = edict()
config.DEBUG.DEBUG = False

def _update_dict(k, v):
    if k == 'DATASET':
        if 'MEAN' in v and v['MEAN']:
            v['MEAN'] = np.array([eval(x) if isinstance(x, str) else x
                                  for x in v['MEAN']])
        if 'STD' in v and v['STD']:
            v['STD'] = np.array([eval(x) if isinstance(x, str) else x
                                 for x in v['STD']])
    if k == 'MODEL':
        if 'EXTRA' in v and 'HEATMAP_SIZE' in v['EXTRA']:
            if isinstance(v['EXTRA']['HEATMAP_SIZE'], int):
                v['EXTRA']['HEATMAP_SIZE'] = np.array(
                    [v['EXTRA']['HEATMAP_SIZE'], v['EXTRA']['HEATMAP_SIZE']])
            else:
                v['EXTRA']['HEATMAP_SIZE'] = np.array(
                    v['EXTRA']['HEATMAP_SIZE'])
        if 'IMAGE_SIZE' in v:
            if isinstance(v['IMAGE_SIZE'], int):
                v['IMAGE_SIZE'] = np.array([v['IMAGE_SIZE'], v['IMAGE_SIZE']])
            else:
                v['IMAGE_SIZE'] = np.array(v['IMAGE_SIZE'])
    for vk, vv in v.items():
        if vk in config[k]:
            config[k][vk] = vv
        else:
            raise ValueError("{}.{} not exist in config.py".format(k, vk))

def update_config(config_file):
    exp_config = None
    with open(config_file) as f:
        exp_config = edict(yaml.load(f))
        for k, v in exp_config.items():
            if k in config:
                if isinstance(v, dict):
                    _update_dict(k, v)
                else:
                    if k == 'SCALES':
                        config[k][0] = (tuple(v))
                    else:
                        config[k] = v
            else:
                raise ValueError("{} not exist in config.py".format(k))


def gen_config(config_file):
    cfg = dict(config)
    for k, v in cfg.items():
        if isinstance(v, edict):
            cfg[k] = dict(v)

    with open(config_file, 'w') as f:
        yaml.dump(dict(cfg), f, default_flow_style=False)


def update_dir(model_dir, log_dir, data_dir):
    if model_dir:
        config.OUTPUT_DIR = model_dir

    if log_dir:
        config.LOG_DIR = log_dir

    if data_dir:
        config.DATA_DIR = data_dir

    config.DATASET.ROOT = os.path.join(
            config.DATA_DIR, config.DATASET.ROOT)

    config.TEST.COCO_BBOX_FILE = os.path.join(
            config.DATA_DIR, config.TEST.COCO_BBOX_FILE)

    config.MODEL.PRETRAINED = os.path.join(
            config.DATA_DIR, config.MODEL.PRETRAINED)


def get_model_name(cfg):
    name = cfg.MODEL.NAME
    full_name = cfg.MODEL.NAME
    return name, full_name


if __name__ == '__main__':
    import sys
    gen_config(sys.argv[1])
