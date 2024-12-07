import os
import cv2
import numpy as np
import torch
from torch.nn import functional as F

from ..base_model import BaseModel
from ..modules import SetBlockWrapper, HorizontalPoolingPyramid, PackSequenceWrapper, SeparateFCs, SeparateBNNecks, CALayers, CALayersP


class XGait(BaseModel):

    def build_network(self, model_cfg):
        # backbone for silhouette
        self.Backbone_sil = self.get_backbone(model_cfg['backbone_cfg'])
        self.Backbone_sil = SetBlockWrapper(self.Backbone_sil)

        # backbone for parsing
        self.Backbone_par = self.get_backbone(model_cfg['backbone_cfg'])
        self.Backbone_par = SetBlockWrapper(self.Backbone_par)

        # Global Cross-granularity Alignment Module
        self.gcm = CALayers(**model_cfg['CALayers'])
        
        # Part Cross-granularity Alignment Module
        self.pcm_up = CALayersP(**model_cfg['CALayers_local'], choosed_part='up')
        self.pcm_middle = CALayersP(**model_cfg['CALayers_local'], choosed_part='middle')
        self.pcm_down = CALayersP(**model_cfg['CALayers_local'], choosed_part='down')
        
        # FCs
        self.FCs_sil = SeparateFCs(**model_cfg['SeparateFCs'])
        self.FCs_par = SeparateFCs(**model_cfg['SeparateFCs'])
        self.FCs_gcm = SeparateFCs(**model_cfg['SeparateFCs'])
        self.FCs_pcm = SeparateFCs(**model_cfg['SeparateFCs'])

        # BNNecks
        self.BNNecks_sil = SeparateBNNecks(**model_cfg['SeparateBNNecks'])
        self.BNNecks_par = SeparateBNNecks(**model_cfg['SeparateBNNecks'])
        self.BNNecks_gcm = SeparateBNNecks(**model_cfg['SeparateBNNecks'])
        self.BNNecks_pcm = SeparateBNNecks(**model_cfg['SeparateBNNecks'])

        self.TP = PackSequenceWrapper(torch.max)
        self.HPP = HorizontalPoolingPyramid(bin_num=model_cfg['bin_num'])

    def forward(self, inputs):
        ipts, labs, _, _, seqL = inputs

        pars = ipts[0]
        sils = ipts[1]
        if len(pars.size()) == 4 and len(sils.size()) == 4:
            pars = pars.unsqueeze(1)
            sils = sils.unsqueeze(1)
            vis_channel = 1
        else:
            raise ValueError("Please check the shape of sils and pars!!!") 

        del ipts
        outs_sil = self.Backbone_sil(sils)  # [n, c, s, h, w]
        outs_par = self.Backbone_par(pars)  # [n, c, s, h, w]

        # Global Cross-granularity Alignment
        outs_gcm = self.gcm(outs_sil, outs_par)  # [n, c, s, h, w]

        # Part Cross-granularity Alignment
        # mask_resize: [n, s, h, w]
        n, c, s, h, w = outs_sil.size()
        mask_resize = F.interpolate(input=pars.squeeze(1), size=(h, w), mode='nearest')
        mask_resize = mask_resize.view(n*s, h, w)

        outs_pcm_up = self.pcm_up(outs_sil, outs_par, mask_resize)  # [n, c, s, h/4, w]
        outs_pcm_middle = self.pcm_middle(outs_sil, outs_par, mask_resize)  # [n, c, s, h/2, w]
        outs_pcm_down = self.pcm_down(outs_sil, outs_par, mask_resize)  # [n, c, s, h/4, w]
        outs_pcm = torch.cat((outs_pcm_up, outs_pcm_middle, outs_pcm_down), dim=-2)  # [n, c, s, h, w]

        # Temporal Pooling, TP
        outs_sil = self.TP(outs_sil, seqL, options={"dim": 2})[0]  # [n, c, h, w]
        outs_par = self.TP(outs_par, seqL, options={"dim": 2})[0]  # [n, c, h, w]
        outs_gcm = self.TP(outs_gcm, seqL, options={"dim": 2})[0]  # [n, c, h, w]
        outs_pcm = self.TP(outs_pcm, seqL, options={"dim": 2})[0]  # [n, c, h, w]

        # Horizontal Pooling Matching, HPM
        feat_sil = self.HPP(outs_sil)  # [n, c, p]
        feat_par = self.HPP(outs_par)  # [n, c, p]
        feat_gcm = self.HPP(outs_gcm)  # [n, c, p]
        feat_pcm = self.HPP(outs_pcm)  # [n, c, p]

        # silhouette part features
        embed_sil = self.FCs_sil(feat_sil)  # [n, c, p]
        _, logits_sil = self.BNNecks_sil(embed_sil)  # [n, c, p]

        # parsing part features
        embed_par = self.FCs_par(feat_par)  # [n, c, p]
        _, logits_par = self.BNNecks_par(embed_par)  # [n, c, p]

        # gcm part features
        embed_gcm = self.FCs_gcm(feat_gcm)  # [n, c, p]
        _, logits_gcm = self.BNNecks_gcm(embed_gcm)  # [n, c, p]

        # pcm part features
        embed_pcm = self.FCs_pcm(feat_pcm)  # [n, c, p]
        _, logits_pcm = self.BNNecks_pcm(embed_pcm)  # [n, c, p]

        # concatenate four parts features
        embed_cat = torch.cat((embed_sil, embed_par, embed_gcm, embed_pcm), dim=-1)
        logits_cat = torch.cat((logits_sil, logits_par, logits_gcm, logits_pcm), dim=-1)

        embed = embed_cat

        n, _, s, h, w = sils.size()
        retval = {
            'training_feat': {
                'triplet': {'embeddings': embed_cat, 'labels': labs},
                'softmax': {'logits': logits_cat, 'labels': labs}
            },
            'visual_summary': {
                'image/sils': sils.reshape(n*s, vis_channel, h, w)
            },
            'inference_feat': {
                'embeddings': embed
            }
        }
        return retval
