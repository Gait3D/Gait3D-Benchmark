# -*- coding: utf-8 -*-
"""
   File Name：     smplgait
   Author :       jinkai Zheng
   E-mail:        zhengjinkai3@hdu.edu.cn
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from ..base_model import BaseModel
from ..modules import SetBlockWrapper, HorizontalPoolingPyramid, PackSequenceWrapper, SeparateFCs, SeparateBNNecks


class SMPLGait_64pixel(BaseModel):
    def __init__(self, cfgs, is_training):
        super().__init__(cfgs, is_training)

    def build_network(self, model_cfg):
        # Baseline
        self.Backbone = self.get_backbone(model_cfg['backbone_cfg'])
        self.Backbone = SetBlockWrapper(self.Backbone)
        self.FCs = SeparateFCs(**model_cfg['SeparateFCs'])
        self.BNNecks = SeparateBNNecks(**model_cfg['SeparateBNNecks'])
        self.TP = PackSequenceWrapper(torch.max)
        self.HPP = HorizontalPoolingPyramid(bin_num=model_cfg['bin_num'])

        # for SMPL
        self.fc1 = nn.Linear(85, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 256)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(p=0.2)
        self.dropout3 = nn.Dropout(p=0.2)
        # self.relu = nn.ReLU()

    def forward(self, inputs):
        ipts, labs, _, _, seqL = inputs

        sils = ipts[0][0]    # [n, s, h, w]
        smpls = ipts[1][0]   # [n, s, h, w]

        # extract SMPL features
        n, s, d = smpls.size()
        sps = smpls.view(-1, d)
        del smpls

        sps = F.relu(self.bn1(self.fc1(sps)))
        sps = F.relu(self.bn2(self.dropout2(self.fc2(sps))))  # (B, 256) or (n, c)
        sps = F.relu(self.bn3(self.dropout3(self.fc3(sps))))  # (B, 256) or (n, c)
        sps = sps.reshape(n*s, 16, 16)
        iden = Variable(torch.eye(16)).unsqueeze(0).repeat(n*s, 1, 1)   # [n*s, 16, 16]
        if sps.is_cuda:
            iden = iden.cuda()
        sps_trans = sps + iden   # [n*s, 16, 16]

        if len(sils.size()) == 4:
            sils = sils.unsqueeze(2)

        del ipts
        outs = self.Backbone(sils)  # [n, s, c, h, w]
        outs_n, outs_s, outs_c, outs_h, outs_w = outs.size()

        zero_tensor = Variable(torch.zeros((outs_n, outs_s, outs_c, outs_h, outs_h-outs_w)))
        if outs.is_cuda:
            zero_tensor = zero_tensor.cuda()
        outs = torch.cat([outs, zero_tensor], -1)    # [n, s, c, h, h]  [n, s, c, 16, 16]
        outs = outs.reshape(outs_n * outs_s * outs_c, outs_h, outs_h)   # [n*s*c, 16, 16]

        sps = sps_trans.unsqueeze(1).repeat(1, outs_c, 1, 1).reshape(outs_n * outs_s * outs_c, 16, 16)

        outs_trans = torch.bmm(outs, sps)
        outs_trans = outs_trans.reshape(outs_n, outs_s, outs_c, outs_h, outs_h)

        # Temporal Pooling, TP
        outs_trans = self.TP(outs_trans, seqL, dim=1)[0]  # [n, c, h, w]
        # Horizontal Pooling Matching, HPM
        feat = self.HPP(outs_trans)  # [n, c, p]
        feat = feat.permute(2, 0, 1).contiguous()  # [p, n, c]
        embed_1 = self.FCs(feat)  # [p, n, c]

        embed_2, logits = self.BNNecks(embed_1)  # [p+1, n, c]

        embed_1 = embed_1.permute(1, 0, 2).contiguous()  # [n, p+1, c]
        logits = logits.permute(1, 0, 2).contiguous()  # [n, p+1, c]  p为part

        n, s, _, h, w = sils.size()
        retval = {
            'training_feat': {
                'triplet': {'embeddings': embed_1, 'labels': labs},
                'softmax': {'logits': logits, 'labels': labs}
            },
            'visual_summary': {
                'image/sils': sils.view(n*s, 1, h, w)
            },
            'inference_feat': {
                'embeddings': embed_1
            }
        }
        return retval


class SMPLGait_128pixel(BaseModel):
    def __init__(self, cfgs, is_training):
        super().__init__(cfgs, is_training)

    def build_network(self, model_cfg):
        # Baseline
        self.Backbone = self.get_backbone(model_cfg['backbone_cfg'])
        self.Backbone = SetBlockWrapper(self.Backbone)
        self.FCs = SeparateFCs(**model_cfg['SeparateFCs'])
        self.BNNecks = SeparateBNNecks(**model_cfg['SeparateBNNecks'])
        self.TP = PackSequenceWrapper(torch.max)
        self.HPP = HorizontalPoolingPyramid(bin_num=model_cfg['bin_num'])

        # for SMPL
        self.fc1 = nn.Linear(85, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 1024)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(1024)
        self.dropout2 = nn.Dropout(p=0.2)
        self.dropout3 = nn.Dropout(p=0.2)
        # self.relu = nn.ReLU()

    def forward(self, inputs):
        ipts, labs, _, _, seqL = inputs

        sils = ipts[0][0]  # [n, s, h, w]
        smpls = ipts[1][0]  # [n, s, d, np]

        # extract SMPL features
        n, s, d = smpls.size()
        sps = smpls.view(-1, d)
        del smpls

        sps = F.relu(self.bn1(self.fc1(sps)))
        sps = F.relu(self.bn2(self.dropout2(self.fc2(sps))))  # (B, 256) or (n, c)
        sps = F.relu(self.bn3(self.dropout3(self.fc3(sps))))  # (B, 1024) or (n, c)
        sps = sps.reshape(n * s, 32, 32)
        iden = Variable(torch.eye(32)).unsqueeze(0).repeat(n * s, 1, 1)  # [n*s, 32, 32]
        if sps.is_cuda:
            iden = iden.cuda()
        sps_trans = sps + iden  # [n*s, 32, 32]

        if len(sils.size()) == 4:
            sils = sils.unsqueeze(2)

        del ipts
        outs = self.Backbone(sils)  # [n, s, c, h, w]
        outs_n, outs_s, outs_c, outs_h, outs_w = outs.size()

        zero_tensor = Variable(torch.zeros((outs_n, outs_s, outs_c, outs_h, outs_h - outs_w)))
        if outs.is_cuda:
            zero_tensor = zero_tensor.cuda()
        outs = torch.cat([outs, zero_tensor], -1)  # [n, s, c, h, h]  [n, s, c, 32, 32]
        outs = outs.reshape(outs_n * outs_s * outs_c, outs_h, outs_h)  # [n*s*c, 32, 32]

        sps = sps_trans.unsqueeze(1).repeat(1, outs_c, 1, 1).reshape(outs_n * outs_s * outs_c, 32, 32)

        outs_trans = torch.bmm(outs, sps)
        outs_trans = outs_trans.reshape(outs_n, outs_s, outs_c, outs_h, outs_h)

        # Temporal Pooling, TP
        outs_trans = self.TP(outs_trans, seqL, dim=1)[0]  # [n, c, h, w]
        # Horizontal Pooling Matching, HPM
        feat = self.HPP(outs_trans)  # [n, c, p]
        feat = feat.permute(2, 0, 1).contiguous()  # [p, n, c]
        embed_1 = self.FCs(feat)  # [p, n, c]

        embed_2, logits = self.BNNecks(embed_1)  # [p+1, n, c]

        embed_1 = embed_1.permute(1, 0, 2).contiguous()  # [n, p+1, c]
        logits = logits.permute(1, 0, 2).contiguous()  # [n, p+1, c]  p is part

        n, s, _, h, w = sils.size()
        retval = {
            'training_feat': {
                'triplet': {'embeddings': embed_1, 'labels': labs},
                'softmax': {'logits': logits, 'labels': labs}
            },
            'visual_summary': {
                'image/sils': sils.view(n * s, 1, h, w)
            },
            'inference_feat': {
                'embeddings': embed_1
            }
        }
        return retval

