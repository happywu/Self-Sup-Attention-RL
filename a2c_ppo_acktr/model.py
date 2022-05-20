# ------------------------------------------------------------------------------
# Copyright (c) by contributors
# Licensed under the MIT License.
# Written by Haiping Wu
# ------------------------------------------------------------------------------
import os
import logging

from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from a2c_ppo_acktr.distributions import Bernoulli, Categorical, DiagGaussian
from a2c_ppo_acktr.utils import init
from .loss import SaliencyMapMSELoss


BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, base=None, base_kwargs=None):
        super(Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}
        if base is None:
            if len(obs_shape) == 3:
                base = CNNBase
            elif len(obs_shape) == 1:
                base = MLPBase
            else:
                raise NotImplementedError

        self.base = base(obs_shape[0], **base_kwargs)

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "MultiBinary":
            num_outputs = action_space.shape[0]
            self.dist = Bernoulli(self.base.output_size, num_outputs)
        else:
            raise NotImplementedError

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, masks, deterministic=False, output_mask=False, output_feat=False,
            ):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks, output_mask=output_mask,
                                                   output_feat=output_feat)

        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):
        value, _, _ = self.base(inputs, rnn_hxs, masks)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs

    def _train_selfsup_attention(self, image_x, image_y, config):
        return self.base._train_selfsup_attention(image_x, image_y, config)


class NNBase(nn.Module):
    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            self.gru = nn.GRU(recurrent_input_size, hidden_size)
            for name, param in self.gru.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0) \
                         .any(dim=-1)
                         .nonzero()
                         .squeeze()
                         .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, hxs = self.gru(
                    x[start_idx:end_idx],
                    hxs * masks[start_idx].view(1, -1, 1))

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)
            hxs = hxs.squeeze(0)

        return x, hxs


class CNNBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=512,
                 feat_from_selfsup_attention=False,
                 feat_add_selfsup_attention=False,
                 feat_mul_selfsup_attention_mask=False,
                 selfsup_attention_num_keypoints=7,
                 selfsup_attention_gauss_std=0.1,
                 selfsup_attention_fix=True,
                 selfsup_attention_fix_keypointer=False,
                 selfsup_attention_pretrain='',
                 selfsup_attention_keyp_maps_pool=False,
                 selfsup_attention_image_feat_only=False,
                 selfsup_attention_feat_masked=False,
                 selfsup_attention_feat_masked_residual=False,
                 selfsup_attention_feat_load_pretrained=True,
                 use_layer_norm=False,
                 selfsup_attention_keyp_cls_agnostic=False,
                 selfsup_attention_feat_use_ln=True,
                 feat_mul_selfsup_attention_mask_residual=True,
                 bottom_up_form_objects=False,
                 bottom_up_form_num_of_objects=10,
                 gaussian_std=0.1,
                 train_selfsup_attention=False,
                 block_selfsup_attention_grad=True,
                 sep_bg_fg_feat=False,
                 mask_threshold=-1.,
                 selfsup_attention_use_instance_norm=False,
                 fix_feature=False
                 ):
        super(CNNBase, self).__init__(recurrent, hidden_size, hidden_size)

        self.feat_mul_selfsup_attention_mask = feat_mul_selfsup_attention_mask
        self.feat_from_selfsup_attention = feat_from_selfsup_attention
        self.feat_add_selfsup_attention = feat_add_selfsup_attention
        self.selfsup_attention_num_keypoints = selfsup_attention_num_keypoints
        self.selfsup_attention_gauss_std = selfsup_attention_gauss_std
        self.selfsup_attention_pretrain = selfsup_attention_pretrain
        self.selfsup_attention_fix = selfsup_attention_fix
        self.selfsup_attention_fix_keypointer = selfsup_attention_fix_keypointer
        self.selfsup_attention_keyp_maps_pool = selfsup_attention_keyp_maps_pool
        self.selfsup_attention_image_feat_only = selfsup_attention_image_feat_only
        self.selfsup_attention_feat_masked = selfsup_attention_feat_masked
        self.selfsup_attention_feat_masked_residual = selfsup_attention_feat_masked_residual
        self.selfsup_attention_feat_load_pretrained = selfsup_attention_feat_load_pretrained
        self.use_layer_norm = use_layer_norm
        self.selfsup_attention_keyp_cls_agnostic = selfsup_attention_keyp_cls_agnostic
        self.selfsup_attention_feat_use_ln = selfsup_attention_feat_use_ln
        self.feat_mul_selfsup_attention_mask_residual = feat_mul_selfsup_attention_mask_residual
        self.bottom_up_form_objects = bottom_up_form_objects
        self.bottom_up_form_num_of_objects = bottom_up_form_num_of_objects
        self.gaussian_std = gaussian_std
        self.train_selfsup_attention = train_selfsup_attention
        self.block_selfsup_attention_grad = block_selfsup_attention_grad
        self.sep_bg_fg_feat = sep_bg_fg_feat
        self.mask_threshold = mask_threshold
        self.selfsup_attention_use_instance_norm = selfsup_attention_use_instance_norm
        self.fix_feature = fix_feature

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        if self.feat_from_selfsup_attention or self.feat_add_selfsup_attention:
            self._feat_encoder = FeatureEncoder(num_inputs=1,
                                                use_layer_norm=self.selfsup_attention_feat_use_ln,
                                                use_instance_norm=self.selfsup_attention_use_instance_norm)
            keypoint_encoder = FeatureEncoder(num_inputs=1,
                                              use_instance_norm=self.selfsup_attention_use_instance_norm,
                                              use_layer_norm=not self.selfsup_attention_use_instance_norm
                                              )

            self._keypointer = Keypointer(in_channels=32, num_keypoints=self.selfsup_attention_num_keypoints,
                                          gauss_std=self.selfsup_attention_gauss_std,
                                          keypoint_encoder=keypoint_encoder,
                                          class_agnostic=self.selfsup_attention_keyp_cls_agnostic)
            if self.train_selfsup_attention:
                assert not self.selfsup_attention_fix, 'selfsup_attention_fix should be False if training needed'
                assert not self.selfsup_attention_fix_keypointer, \
                    'selfsup_attention_fix_keypointer should be False if training needed'
                decoder = Decoder(in_channels=32, out_channels=1,
                                  use_layer_norm=self.selfsup_attention_feat_use_ln,
                                  use_instance_norm=self.selfsup_attention_use_instance_norm,
                                  with_sigmoid=False)
                self.selfsup_attention_criterion = SaliencyMapMSELoss(use_target_weight=False)
                self.selfsup_attention = SelfSupAttention(self._feat_encoder, self._keypointer, decoder)

            if os.path.exists(self.selfsup_attention_pretrain):
                selfsup_attention_state_dict = torch.load(self.selfsup_attention_pretrain)
                logger.info('=> loading pretrained model {}'.format(self.selfsup_attention_pretrain))

                feat_encoder_state_dict = OrderedDict((_k.replace('_feature_encoder.', '').
                                                       replace('_feat_encoder.', '')[5 if 'base.' in _k else 0:]
                                                       , _v)
                                                      for _k, _v in selfsup_attention_state_dict.items() \
                                                      if ('_feature_encoder' in _k or '_feat_encoder' in _k) \
                                                      and 'selfsup_attention' not in _k)
                keypointer_state_dict = OrderedDict((_k.replace('_keypointer.', '')[5 if 'base.' in _k else 0:], _v)
                                                    for _k, _v in selfsup_attention_state_dict.items() \
                                                    if '_keypointer' in _k and 'selfsup_attention' not in _k)
                if self.selfsup_attention_feat_load_pretrained:
                    self._feat_encoder.load_state_dict(feat_encoder_state_dict, strict=True)
                self._keypointer.load_state_dict(keypointer_state_dict, strict=True)

                if self.train_selfsup_attention:
                    self.selfsup_attention.load_state_dict(selfsup_attention_state_dict, strict=True)

                # fix weights
                if self.selfsup_attention_fix:
                    for param in self._feat_encoder.parameters():
                        param.requires_grad = False
                    for param in self._keypointer.parameters():
                        param.requires_grad = False
                else:
                    if self.selfsup_attention_fix_keypointer:
                        for param in self._keypointer.parameters():
                            param.requires_grad = False

        feat_channels = 32
        if not self.feat_from_selfsup_attention:
            self.convs_1 = nn.Sequential(
                init_(nn.Conv2d(num_inputs, 32, 8, stride=4)),
                nn.ReLU())
            self.convs_2 = nn.Sequential(
                init_(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU())
            self.convs_3 = nn.Sequential(
                init_(nn.Conv2d(64, 32, 3, stride=1)), nn.ReLU())

        self.fc = nn.Sequential(
            Flatten(),
            init_(nn.Linear(feat_channels * 7 * 7, hidden_size)), nn.ReLU())

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.critic_linear = init_(nn.Linear(hidden_size, 1))
        self.train()

    def forward(self, inputs, rnn_hxs, masks, output_mask=False,
                output_feat=False,
                ):
        inputs_normalized = inputs / 255.0

        meta = {}
        original_inputs_normalized = inputs_normalized
        meta['input'] = inputs_normalized[:, -1:]

        if self.feat_from_selfsup_attention or self.feat_add_selfsup_attention:
            inputs_frame = inputs_normalized[:, -1:]
            feat = self._feat_encoder(inputs_frame)  # (batch_size x 4) x 64 x h x w
            keypoints_centers, keypoints_maps = self._keypointer(inputs_frame)
            if self.block_selfsup_attention_grad:
                feat = feat.detach()
                keypoints_maps = keypoints_maps.detach()
                if keypoints_centers is not None:
                    keypoints_centers = keypoints_centers.detach()
            if self.selfsup_attention_keyp_maps_pool:
                keypoints_maps = keypoints_maps.max(dim=1, keepdim=True)[0]
            meta['keypoints_maps'] = keypoints_maps

        if not self.feat_from_selfsup_attention:
            x = self.convs_1(inputs_normalized)
            x = self.convs_2(x)
            x = self.convs_3(x)

            if self.feat_add_selfsup_attention:
                if self.feat_mul_selfsup_attention_mask:
                    if self.feat_mul_selfsup_attention_mask_residual:
                        x = x * (1 + keypoints_maps)
                    else:
                        x = x * keypoints_maps
        else:
            x = feat
            if self.feat_mul_selfsup_attention_mask:
                if self.feat_mul_selfsup_attention_mask_residual:
                    x = x * (1 + keypoints_maps)
                else:
                    x = x * keypoints_maps

        if self.fix_feature:
            x = x.detach()

        x = self.fc(x)

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        if output_mask:
            rnn_hxs = [rnn_hxs, mask]
        if output_feat:
            if self.output_original_mask:
                rnn_hxs = [rnn_hxs, ori_feat]
            else:
                rnn_hxs = [rnn_hxs, saliency_map]
        else:
            rnn_hxs = [rnn_hxs, meta]
        return self.critic_linear(x), x, rnn_hxs

    def _get_keypoints_feat(self, image_feat, kpts_mask):
        feats = []
        num_keypoints = kpts_mask.shape[1]
        for k in range(num_keypoints):
            # n x 1 x h x w
            mask = kpts_mask[:, k:k+1]
            kpt_feat = mask * image_feat  # n x c x h x w
            kpt_feat = F.adaptive_avg_pool2d(kpt_feat, 1)  # global avg pool
            feats.append(kpt_feat.view(kpt_feat.shape[0], kpt_feat.shape[1]))

        # n x k x c
        feats = torch.stack(feats, dim=1)
        return feats

    def _get_guassian_maps(self, mu, map_size, inv_std, power=2):
        mu_y, mu_x = mu[:, :, 0:1], mu[:, :, 1:2]
        y = torch.arange(map_size[0], device=mu.device, dtype=mu.dtype)
        x = torch.arange(map_size[1], device=mu.device, dtype=mu.dtype)
        y = y / map_size[0] * 2 - 1
        x = x / map_size[1] * 2 - 1
        y = y.view((1, 1, map_size[0], 1))
        x = x.view((1, 1, 1, map_size[1]))
        mu_y = mu_y.unsqueeze(dim=-1)
        mu_x = mu_x.unsqueeze(dim=-1)

        g_y = (y - mu_y) ** power
        g_x = (x - mu_x) ** power
        dist = (g_y + g_x) * (inv_std ** power)
        g_yx = torch.exp(-dist)
        return g_yx

    def _get_feat_from_objects_loc(self, image_feat, keypoints_loc, return_mask=False):
        map_size = image_feat.shape[2:]
        kpts_mask = self._get_guassian_maps(keypoints_loc, map_size, 1.0 / self.gaussian_std)
        feats = []
        num_keypoints = kpts_mask.shape[1]
        for k in range(num_keypoints):
            # n x 1 x h x w
            mask = kpts_mask[:, k:k + 1]
            kpt_feat = mask * image_feat  # n x c x h x w
            kpt_feat = F.adaptive_avg_pool2d(kpt_feat, 1)  # global avg pool
            feats.append(kpt_feat.view(kpt_feat.shape[0], kpt_feat.shape[1]))

        # n x k x c
        feats = torch.stack(feats, dim=1)
        if return_mask:
            return feats, kpts_mask
        return feats

    def _get_keypoints_loc_from_mask(self, mask):
        if not hasattr(self, 'parser'):
            self.parser = HeatmapParser(max_objects=self.bottom_up_form_num_of_objects)
        if not hasattr(self, 'smooth_layer'):
            self.smooth_layer = GaussianLayer().cuda()
        mask = self.smooth_layer(mask)
        keypoints = self.parser.parse(mask)
        keypoints_loc = keypoints['loc_k']
        keypoints_loc = keypoints_loc.squeeze(dim=1)
        return keypoints_loc

    def _get_keypoints_feat_bottomup(self, image_feat, kpts_mask, return_mask=False):
        keypoints_loc = self._get_keypoints_loc_from_mask(kpts_mask)
        # batch_size x num_keypoints x 2 (x, y)
        feats = self._get_feat_from_objects_loc(image_feat, keypoints_loc, return_mask=return_mask)
        return feats, keypoints_loc

    def _get_pixel_feat_with_loc(self, x):
        # x : n x c x h x w
        n, c, h, w = x.shape
        feat = x.view(n, c, h * w)
        feat = feat.transpose(1, 2)  # n x (hxw) x c
        loc_y = torch.arange(h,device=x.device, dtype=x.dtype).unsqueeze(1).expand(h, w)
        loc_x = torch.arange(w,device=x.device, dtype=x.dtype).unsqueeze(0).expand(h, w)
        loc_y = (loc_y / (h-1)) * 2. - 1
        loc_x = (loc_x / (w-1)) * 2. - 1
        coords = torch.stack((loc_y, loc_x), dim=2).unsqueeze(0).view(1, h*w, 2).repeat(n, 1, 1) #  n x (h x w) x 2
        feat_with_loc = torch.cat((feat, coords), dim=2)
        return feat_with_loc

    def _relation_net_forward(self, object_feats_with_loc):
        x_i = torch.unsqueeze(object_feats_with_loc, 1)
        num_objects = object_feats_with_loc.shape[1]
        x_i = x_i.repeat(1, num_objects, 1, 1)
        x_j = torch.unsqueeze(object_feats_with_loc, 2)
        x_j = x_j.repeat(1, 1, num_objects, 1)
        x = torch.cat([x_i, x_j], dim=3)
        x = x.view(-1, x.shape[-1])
        return x

    def _train_selfsup_attention(self, input, target, config):
        input = input / 255.
        target = target / 255.
        autoencoder_loss = config.AUTOENCODER_LOSS
        heatmap_sparsity_loss = config.HEATMAP_SPARSITY_LOSS
        meta, image_b_keypoints_maps = self.selfsup_attention(input, target,
                                                              output_heatmaps=True,
                                                              autoencoder_loss=autoencoder_loss)
        output = meta['reconstruct_image_b']
        thre = float(config.RECONSTRUCT_LOSS_THRESHOLD)
        loss = self.selfsup_attention_criterion(output, target, threshold=thre)
        if autoencoder_loss:
            autoencoder_output = meta['reconstruct_image_b_auto']
            auto_loss = self.selfsup_attention_criterion(autoencoder_output, target)
            loss = loss + auto_loss
        if heatmap_sparsity_loss:
            heatmap_loss = image_b_keypoints_maps.mean()
            loss = loss + heatmap_loss * float(config.HEATMAP_SPARSITY_LOSS_WEIGHT)
        return loss, output, image_b_keypoints_maps


class MLPBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=64):
        super(MLPBase, self).__init__(recurrent, num_inputs, hidden_size)

        if recurrent:
            num_inputs = hidden_size

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs


class AutoEncoder(nn.Module):
    def __init__(self, num_inputs):
        super(AutoEncoder, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))
        self.convs_1 = nn.Sequential(
            init_(nn.Conv2d(num_inputs, 32, 8, stride=4)),
            nn.ReLU())
        self.convs_2 = nn.Sequential(
            init_(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU())
        self.convs_3 = nn.Sequential(
            init_(nn.Conv2d(64, 32, 3, stride=1)),
            nn.ReLU())

        self.deconv_1 = nn.Sequential(
            init_(nn.ConvTranspose2d(32, 64, 3, stride=1)),
            nn.ReLU())
        self.deconv_2 = nn.Sequential(
            init_(nn.ConvTranspose2d(64, 32, 4, stride=2)),
            nn.ReLU())
        self.deconv_3 = nn.Sequential(
            init_(nn.ConvTranspose2d(32, 16, 8, stride=4)),
            nn.ReLU())

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('sigmoid'))
        self.final_layer = nn.Sequential(
            init_(nn.Conv2d(16, num_inputs, 1)),
            nn.Sigmoid())

    def forward(self, inputs, back_feat=False, output_feat=False):
        x = self.convs_1(inputs)
        x = self.convs_2(x)
        feat = self.convs_3(x)

        x = self.deconv_1(feat)
        x = self.deconv_2(x)
        x = self.deconv_3(x)
        x = self.final_layer(x)

        if output_feat:
            return [x, feat]
        else:
            return x


# encoder, keypoint encoder and selfsup_attention model
# in 'unsupervised learning of object keypoints for perception and control' paper.
class FeatureEncoder(nn.Module):
    def __init__(self, num_inputs, use_layer_norm=True,
                 use_instance_norm=False):
        super(FeatureEncoder, self).__init__()
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        self.convs_1 = nn.Sequential(
            init_(nn.Conv2d(num_inputs, 32, 8, stride=4)),
            nn.LayerNorm([32, 20, 20]) if use_layer_norm else \
                (nn.InstanceNorm2d(32) if use_instance_norm else nn.Identity()),
            nn.ReLU())
        self.convs_2 = nn.Sequential(
            init_(nn.Conv2d(32, 64, 4, stride=2)),
            nn.LayerNorm([64, 9, 9]) if use_layer_norm else \
                (nn.InstanceNorm2d(64) if use_instance_norm else nn.Identity()),
            nn.ReLU())
        self.convs_3 = nn.Sequential(
            init_(nn.Conv2d(64, 32, 3, stride=1)),
            nn.LayerNorm([32, 7, 7]) if use_layer_norm else \
                (nn.InstanceNorm2d(32) if use_instance_norm else nn.Identity()),
            nn.ReLU())

    def forward(self, inputs):
        x = self.convs_1(inputs)
        x = self.convs_2(x)
        x = self.convs_3(x)
        return x


class Keypointer(nn.Module):

    def __init__(self, in_channels, num_keypoints, gauss_std, keypoint_encoder, class_agnostic=False):
        super(Keypointer, self).__init__()
        self._num_keypoints = num_keypoints
        self._guassian_std = gauss_std
        self._keypoint_encoder = keypoint_encoder
        self._class_agnostic = class_agnostic

        if self._class_agnostic:
            self._num_keypoints = 1
            self.smooth_layer = GaussianLayer()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))
        self.keypoint_conv = init_(nn.Conv2d(in_channels, self._num_keypoints,
                                             1))

    def _heatmap_nms(self, heat, kernel=3):
        pad = (kernel - 1) // 2

        hmax = nn.functional.max_pool2d(
            heat, (kernel, kernel), stride=1, padding=pad)
        keep = (hmax == heat).float()
        return heat * keep

    def _get_keypoint_mu(self, heatmaps):
        # Integral loss to get center coordinates
        # n x c x h x w
        y_dim, x_dim = heatmaps.shape[2:4]
        accu_x = heatmaps.mean(dim=2)
        # n x c x h
        accu_y = heatmaps.mean(dim=3)

        accu_x = accu_x.softmax(axis=-1)
        accu_y = accu_y.softmax(axis=-1)

        accu_x = accu_x * torch.arange(x_dim,device=accu_x.device, dtype=accu_x.dtype)
        accu_y = accu_y * torch.arange(y_dim,device=accu_y.device, dtype=accu_y.dtype)

        accu_x = accu_x.sum(dim=2)
        accu_y = accu_y.sum(dim=2)

        # convert [0, 1] to [-1, 1]
        accu_x = accu_x / x_dim * 2 - 1
        accu_y = accu_y / y_dim * 2 - 1

        # n x c x 2 (y, x)
        gauss_mu = torch.stack([accu_y, accu_x], dim=2)
        return gauss_mu

    def _get_guassian_maps(self, mu, map_size, inv_std, power=2):
        mu_y, mu_x = mu[:, :, 0:1], mu[:, :, 1:2]
        y = torch.arange(map_size[0], device=mu.device, dtype=mu.dtype)
        x = torch.arange(map_size[1], device=mu.device, dtype=mu.dtype)
        y = y / map_size[0] * 2 - 1
        x = x / map_size[1] * 2 - 1
        y = y.view((1, 1, map_size[0], 1))
        x = x.view((1, 1, 1, map_size[1]))
        mu_y = mu_y.unsqueeze(dim=-1)
        mu_x = mu_x.unsqueeze(dim=-1)

        g_y = (y - mu_y) ** power
        g_x = (x - mu_x) ** power
        dist = (g_y + g_x) * (inv_std ** power)
        g_yx = torch.exp(-dist)
        return g_yx

    def _get_guassian_maps_cls_agnostic(self, heatmaps, threshold=0.5, with_sigmoid=True):
        if with_sigmoid:
            heatmaps = F.sigmoid(heatmaps)
        heat = heatmaps
        return heat

    def forward(self, inputs, form_gaussian=True, with_sigmoid=True, return_heatmaps=False):
        feat = self._keypoint_encoder(inputs)
        heatmaps = self.keypoint_conv(feat)

        if self._class_agnostic or not form_gaussian:
            gauss_mu = None
            gauss_maps = self._get_guassian_maps_cls_agnostic(heatmaps, with_sigmoid=with_sigmoid)
        else:
            gauss_mu = self._get_keypoint_mu(heatmaps)
            map_size = heatmaps.shape[2:]
            gauss_maps = self._get_guassian_maps(gauss_mu, map_size, 1.0 / self._guassian_std)

        if return_heatmaps:
            return gauss_mu, gauss_maps, heatmaps
        else:
            return gauss_mu, gauss_maps


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, use_layer_norm=True,
                 with_sigmoid=True,
                 use_instance_norm=False):
        super(Decoder, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))
        init_sigmoid = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('sigmoid'))

        self.deconv_1 = nn.Sequential(
            init_(nn.ConvTranspose2d(32, 64, 3, stride=1)),
            nn.LayerNorm([64, 7, 7]) if use_layer_norm else \
                (nn.InstanceNorm2d(64) if use_instance_norm else nn.Identity()),
            nn.ReLU())
        self.deconv_2 = nn.Sequential(
            init_(nn.ConvTranspose2d(64, 32, 4, stride=2)),
            nn.LayerNorm([32, 9, 9]) if use_layer_norm else \
                (nn.InstanceNorm2d(32) if use_instance_norm else nn.Identity()),
            nn.ReLU())
        self.deconv_3 = nn.Sequential(
            init_(nn.ConvTranspose2d(32, 16, 8, stride=4)),
            nn.LayerNorm([16, 84, 84]) if use_layer_norm else \
                (nn.InstanceNorm2d(16) if use_instance_norm else nn.Identity()),
            nn.ReLU())
        if with_sigmoid:
            self.final_layer = nn.Sequential(
                init_sigmoid(nn.Conv2d(16, out_channels, 3, padding=1, stride=1)),
                nn.Sigmoid())
        else:
            self.final_layer = nn.Sequential(
                init_(nn.Conv2d(16, out_channels, 3, padding=1, stride=1)))

    def forward(self, x, return_mid_feat=False):
        ori_x = x
        if return_mid_feat:
            mid_feat = self.fc(x)
        x = self.deconv_1(x)
        x = self.deconv_2(x)
        x = self.deconv_3(x)
        x = self.final_layer(x)

        if return_mid_feat:
            return x, ori_x
        else:
            return x

import scipy
class GaussianLayer(nn.Module):
    def __init__(self):
        super(GaussianLayer, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(1, 1, 3, stride=1, padding=1, bias=None)
        )

        self.weights_init()
    def forward(self, x):
        return self.seq(x)

    def weights_init(self):
        n= np.zeros((3,3))
        n[1,1] = 1
        k = scipy.ndimage.gaussian_filter(n,sigma=0.5)
        k = k / k[1, 1]
        for name, f in self.named_parameters():
            f.data.copy_(torch.from_numpy(k))
            f.requires_grad = False

class SelfSupAttention(nn.Module):
    def __init__(self, feature_encoder, keypointer, decoder):
        super(SelfSupAttention, self).__init__()
        self._feature_encoder = feature_encoder
        self._keypointer = keypointer
        self._decoder = decoder
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def _get_keypoints_feat(self, image_feat, kpts_mask):
        feats = []
        num_keypoints = kpts_mask.shape[1]
        for k in range(num_keypoints):
            # n x 1 x h x w
            mask = kpts_mask[:, k:k+1]
            kpt_feat = mask * image_feat  # n x c x h x w
            kpt_feat = F.adaptive_avg_pool2d(kpt_feat, 1)  # global avg pool
            feats.append(kpt_feat.view(kpt_feat.shape[0], kpt_feat.shape[1]))

        # n x k x c
        feats = torch.stack(feats, dim=1)
        return feats

    def calc_feat_similarity(self, feat_a, feat_b):
        # n x k x c
        return F.cosine_similarity(feat_a, feat_b, dim=-1)

    def calc_feat_dist(self, feat_a, feat_b):
        # n x feat
        n, k, d = feat_a.shape
        return F.pairwise_distance(feat_a.view(n*k, d), feat_b.view(n*k, d), keepdim=True).view(n, k)

    def _get_keypoints_loc_from_mask(self, mask):
        if not hasattr(self, 'parser'):
            self.parser = HeatmapParser(max_objects=20, threshold=None)
        if not hasattr(self, 'smooth_layer'):
            self.smooth_layer = GaussianLayer().cuda()
        mask = self.smooth_layer(mask)
        keypoints = self.parser.parse(mask)
        keypoints_loc = keypoints['loc_k']
        keypoints_loc = keypoints_loc.squeeze(dim=1)
        keypoints_val = keypoints['val_k'].squeeze(dim=1)
        return keypoints_loc, keypoints_val

    def forward(self, image_a, image_b, output_heatmaps=False, output_keypoints=False,
                output_a_keypoints=False,
                autoencoder_loss=False,
                block_image_feat_gradient=False,
                output_keypoints_loc=False):

        image_a_feat = self._feature_encoder(image_a)
        image_b_feat = self._feature_encoder(image_b)

        image_a_keypoints_centers, image_a_keypoints_maps = self._keypointer(image_a)
        image_b_keypoints_centers, image_b_keypoints_maps, image_b_heatmaps = \
            self._keypointer(image_b, return_heatmaps=True)

        image_a_feat = image_a_feat.detach()
        if block_image_feat_gradient:
            image_b_feat_nonblock = image_b_feat
            image_b_feat = image_b_feat.detach()
        if image_a_keypoints_centers is not None:
            image_a_keypoints_centers = image_a_keypoints_centers.detach()
        image_a_keypoints_maps = image_a_keypoints_maps.detach()

        num_keypoints = image_a_keypoints_maps.shape[1]

        meta = {}
        transported_feat = image_a_feat
        for k in range(num_keypoints):
            # n x h x w
            mask_a = image_a_keypoints_maps[:, k:k+1]
            mask_b = image_b_keypoints_maps[:, k:k+1]
            # suppress features from image a, around both keypoint locations
            transported_feat = (1-mask_a) * (1-mask_b) * transported_feat
            # copy features from image b around keypoints for image b
            transported_feat += mask_b * image_b_feat
            foreground_feat = image_b_feat * mask_b

        if num_keypoints == 1:
            meta['image_b_foreground'] = image_b_feat * image_b_keypoints_maps[:, 0:1]
            meta['image_a_background'] = image_a_feat * (1-image_a_keypoints_maps[:,0:1])

        reconstruct_image_b = self._decoder(transported_feat)
        reconstruct_foreground_b = self._decoder(foreground_feat.detach())

        meta['reconstruct_image_b_foreground'] = reconstruct_foreground_b
        if output_keypoints_loc and image_b_keypoints_maps.shape[1] == 1:
            keypoints_loc, keypoints_val = self._get_keypoints_loc_from_mask(image_b_keypoints_maps)
            meta['keypoints_bu_loc'] = keypoints_loc
            meta['keypoints_bu_val'] = keypoints_val

        meta['keypoints_b_heatmaps'] = image_b_heatmaps
        meta['image_a_keypoints_maps'] = image_a_keypoints_maps
        if autoencoder_loss:
            if block_image_feat_gradient:
                reconstruct_image_b_auto = self._decoder(image_b_feat_nonblock)
            else:
                reconstruct_image_b_auto = self._decoder(image_b_feat)
            meta['reconstruct_image_b_auto'] = reconstruct_image_b_auto
        meta['reconstruct_image_b'] = reconstruct_image_b

        if output_a_keypoints:
            image_b_keypoints_centers = [image_a_keypoints_centers, image_b_keypoints_centers]
        if output_heatmaps and output_keypoints:
            return meta, image_b_keypoints_maps, image_b_keypoints_centers
        elif output_heatmaps:
            return meta, image_b_keypoints_maps
        elif output_keypoints:
            return meta, image_b_keypoints_centers

        return meta


class HeatmapParser(object):

    def __init__(self, nms_kernel=3, nms_padding=1, max_objects=10, threshold=None):
        self.pool = torch.nn.MaxPool2d(
            nms_kernel, 1, nms_padding
        )
        self.max_objects = max_objects
        self.threshold = threshold

    def nms(self, det):
        maxm = self.pool(det)
        maxm = torch.eq(maxm, det).float()
        det = det * maxm
        return det

    def top_k(self, det):
        # det: batch_size x num_keypoints (1 in cls agn mode) x h x w
        # det = det.reshape((1, 1, det.shape[0], det.shape[1]))
        # det = torch.tensor(det, requires_grad=False, dtype=torch.float64)
        det = self.nms(det)
        num_images = det.size(0)
        num_joints = det.size(1)
        h = det.size(2)
        w = det.size(3)
        det = det.view(num_images, num_joints, -1)

        if self.threshold:
            # num_images = 1
            ind = []
            for _det in det.view(num_images, -1):
                _ind = torch.nonzero(_det > self.threshold).flatten()
                _ind_padded = torch.zeros(self.max_objects)
                _ind_padded[:len(_ind)] = _ind
                ind.append(_ind_padded)
            ind = torch.stack(ind, dim=0)
            ind = ind.view(num_images, 1, -1)
            # fake val
            val_k = torch.ones_like(ind)
        else:
            val_k, ind = det.topk(self.max_objects, dim=2)

        x = ind % w
        y = (ind / w).long()

        x = (x.float() / h - 0.5) * 2
        y = (y.float() / w - 0.5) * 2

        ind_k = torch.stack((y, x), dim=3)

        ans = {
            'loc_k': ind_k,
            'val_k': val_k
        }

        return ans

    def parse(self, det):
        ans = self.top_k(det)
        return ans
