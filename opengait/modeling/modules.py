import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utils import clones, is_list_or_tuple
from torchvision.ops import RoIAlign


class HorizontalPoolingPyramid():
    """
        Horizontal Pyramid Matching for Person Re-identification
        Arxiv: https://arxiv.org/abs/1804.05275
        Github: https://github.com/SHI-Labs/Horizontal-Pyramid-Matching
    """

    def __init__(self, bin_num=None):
        if bin_num is None:
            bin_num = [16, 8, 4, 2, 1]
        self.bin_num = bin_num

    def __call__(self, x):
        """
            x  : [n, c, h, w]
            ret: [n, c, p] 
        """
        n, c = x.size()[:2]
        features = []
        for b in self.bin_num:
            z = x.view(n, c, b, -1)
            z = z.mean(-1) + z.max(-1)[0]
            features.append(z)
        return torch.cat(features, -1)


class SetBlockWrapper(nn.Module):
    def __init__(self, forward_block):
        super(SetBlockWrapper, self).__init__()
        self.forward_block = forward_block

    def forward(self, x, *args, **kwargs):
        """
            In  x: [n, c_in, s, h_in, w_in]
            Out x: [n, c_out, s, h_out, w_out]
        """
        n, c, s, h, w = x.size()
        x = self.forward_block(x.transpose(
            1, 2).reshape(-1, c, h, w), *args, **kwargs)
        output_size = x.size()
        return x.reshape(n, s, *output_size[1:]).transpose(1, 2).contiguous()


class SetBlockWrapperseqL(nn.Module):
    def __init__(self, forward_block):
        super(SetBlockWrapperseqL, self).__init__()
        self.forward_block = forward_block

    def forward(self, info, *args, **kwargs):
        x, seqL, dim = info
        n, c, s, h, w = x.size()
        info = (x.transpose(1, 2).reshape(-1, c, h, w), seqL, dim)
        info = self.forward_block(info, *args, **kwargs)
        x, _, _ = info
        output_size = x.size()
        return x.reshape(n, s, *output_size[1:]).transpose(1, 2).contiguous()


class LeakyReLUseqL(nn.Module):
    def __init__(self, relu_func):
        super(LeakyReLUseqL, self).__init__()
        self.relu_func = relu_func

    def forward(self, info):
        seqs, seqL, dim = info
        seqs = self.relu_func(seqs)
        return (seqs, seqL, dim)


class MaxPool2dseqL(nn.Module):
    def __init__(self, pool_func):
        super(MaxPool2dseqL, self).__init__()
        self.pool_func = pool_func

    def forward(self, info):
        seqs, seqL, dim = info
        seqs = self.pool_func(seqs)
        return (seqs, seqL, dim)


class PackSequenceWrapper(nn.Module):
    def __init__(self, pooling_func):
        super(PackSequenceWrapper, self).__init__()
        self.pooling_func = pooling_func

    def forward(self, seqs, seqL, dim=2, options={}):
        """
            In  seqs: [n, c, s, ...]
            Out rets: [n, ...]
        """
        if seqL is None:
            return self.pooling_func(seqs, **options)
        seqL = seqL[0].data.cpu().numpy().tolist()
        start = [0] + np.cumsum(seqL).tolist()[:-1]

        rets = []
        for curr_start, curr_seqL in zip(start, seqL):
            narrowed_seq = seqs.narrow(dim, curr_start, curr_seqL)
            rets.append(self.pooling_func(narrowed_seq, **options))
        if len(rets) > 0 and is_list_or_tuple(rets[0]):
            return [torch.cat([ret[j] for ret in rets])
                    for j in range(len(rets[0]))]
        return torch.cat(rets)


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=False, **kwargs)

    def forward(self, x):
        x = self.conv(x)
        return x


class BasicConv2dseqL(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, **kwargs):
        super(BasicConv2dseqL, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=False, **kwargs)

    def forward(self, info):
        x, seqL, dim = info
        x = self.conv(x)
        return (x, seqL, dim)


class BasicConv2dMTSRES(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, n_segment, n_div, skip_times, shift_type, **kwargs):
        super(BasicConv2dMTSRES, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=False, **kwargs)
        # self.start_iter = start_iter
        self.n_segment = n_segment
        self.fold_div = n_div
        self.skip_times = skip_times
        self.shift_type = shift_type

    def forward(self, info):
        x, seqL, dim = info

        x_shifts = []
        for skip_time in self.skip_times:
            x_s = self.shift(x, seqL, dim, n_segment=self.n_segment, fold_div=self.fold_div,
                             skip_time=skip_time, shift_type=self.shift_type)
            x_s = self.conv(x_s)
            x_shifts.append(x_s)

        # x_shortcut
        x = self.conv(x)

        for x_shift in x_shifts:
            x = x + x_shift
        return (x, seqL, dim)

    def shift(self, x, seqL, seq_dim=1, n_segment=30, fold_div=3, skip_time=1, shift_type="bi_direction"):
        nt, c, h, w = x.size()

        if seqL is None:
            n_batch = nt // n_segment
            x_v = x.view(n_batch, n_segment, c, h, w)
            out = self.shift_unit(x_v, c, fold_div, skip_time, shift_type)
        else:
            seqL = seqL[0].data.cpu().numpy().tolist()
            start = [0] + np.cumsum(seqL).tolist()[:-1]

            rets = []
            for curr_start, curr_seqL in zip(start, seqL):
                narrowed_x = x.narrow(seq_dim, curr_start, curr_seqL)
                narrowed_x = narrowed_x.unsqueeze(0)
                rets.append(self.shift_unit(narrowed_x, c, fold_div, skip_time, shift_type))
            out = torch.cat(rets, 1)
        return out.view(nt, c, h, w)

    def shift_unit(self, x, c, fold_div, skip_time, shift_type):
        # x: (B, T, C, H, W)
        fold = c // fold_div
        out = torch.zeros_like(x)
        if shift_type == "uni_direction":
            out[:, skip_time:, :fold] = x[:, :skip_time * -1, :fold]  # shift right 1
            out[:, skip_time:, -1 * fold:] = x[:, :skip_time * -1, -1 * fold:]  # shift right 2
            out[:, :, fold: -1 * fold] = x[:, :, fold: -1 * fold]  # not shift
        elif shift_type == "bi_direction":
            out[:, :skip_time * -1, :fold] = x[:, skip_time:, :fold]  # shift left 1
            out[:, skip_time:, -1 * fold:] = x[:, :skip_time * -1, -1 * fold:]  # shift right 2
            out[:, :, fold: -1 * fold] = x[:, :, fold: -1 * fold]  # not shift
        else:
            raise KeyError(f"The shift_type should be in [uni_direction, bi_direction], but got {shift_type}")
        return out


class CALayers(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CALayers, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels * 2, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels * 2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, y):
        """
            x: [n, c, s, h, w]
            y: [n, c, s, h, w]
        """
        n, c, s, h, w = x.size()
        x_ = x.transpose(1, 2).reshape(-1, c, h, w)   # [n*s, c, h, w]
        y_ = y.transpose(1, 2).reshape(-1, c, h, w)   # [n*s, c, h, w]
        x_ = self.avg_pool(x_).view(n*s, c)
        y_ = self.avg_pool(y_).view(n*s, c)
        z = torch.cat([x_, y_], dim=1)     # [n*s, 2*c]
        z = self.fc(z).view(n, s, 2*c, 1, 1).transpose(1, 2)  # [n, 2*c, s, 1, 1]
        return x * z[:,:c,:,:,:].expand_as(x) + y * z[:,c:,:,:,:].expand_as(y)


def PartPooling(x, with_max_pool=True):
    """
        Part Pooling for GCN
        x   : [n*s, c, h, w]
        ret : [n*s, c] 
    """
    n_s, c, h, w = x.size()
    z = x.view(n_s, c, -1)  # [n, p, c, h*w]
    if with_max_pool:
        z = z.mean(-1) + z.max(-1)[0]   # [n*s, c]
    else:
        z = z.mean(-1)   # [n*s, c]
    return z


class CALayersP(nn.Module):
    def __init__(self, channels, reduction=16, with_max_pool=True, choosed_part=''):
        super(CALayersP, self).__init__()
        self.choosed_part = choosed_part
        self.with_max_pool = with_max_pool
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.gammas = torch.nn.Parameter(torch.ones(1) * 0.75)
        self.fc = nn.Sequential(
            nn.Linear(channels * 2, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels * 2, bias=False),
            nn.Sigmoid()
        )

    def learnable_division(self, z, mask_resize, choosed_part=''):
        """
            z: [n*s, c, h, w]
            mask_resize: [n*s, h, w]
            return [n*s, c, h, w]
            ***Coarse Parts:
            up:     [1]  Head
            middle: [2, 3, 4, 5, 6, 11]  Torso, Left-arm, Left-hand, Right-arm, Right-hand, Dress
            down:   [7, 8, 9, 10, 11]    Left-leg, Left-foot, Right-leg, Right-foot, Dress
        """
        split_parts = {
            'up': [1],
            'middle': [2, 3, 4, 5, 6, 11],
            'down': [7, 8, 9, 10, 11],
        }
        choosed_part_list = split_parts[choosed_part]
        choosed_part_mask = mask_resize.long() == -1
        for part_i in choosed_part_list:
            choosed_part_mask += (mask_resize.long() == part_i)

        mask = choosed_part_mask.unsqueeze(1)
        z_feat = mask.float() * z * self.gammas + (~mask).float() * z * (1.0 - self.gammas)   # [n*s, c, h, w]

        return z_feat
    
    def forward(self, x, y, mask_resize):
        """
            x: [n, c, s, h, w]
            y: [n, c, s, h, w]
        """
        n, c, s, h, w = x.size()
        h_split = h // 4
        x_ = x.transpose(1, 2).reshape(-1, c, h, w)   # [n*s, c, h, w]
        y_ = y.transpose(1, 2).reshape(-1, c, h, w)   # [n*s, c, h, w]

        x_ = x_[:, :, :h_split, :]   # [n*s, c, h/4, w]
        x_ = self.avg_pool(x_).view(n*s, c) + self.max_pool(x_).view(n*s, c)

        y_ = self.learnable_division(y_, mask_resize, self.choosed_part)  # [n*s, c, h, w]
        y_ = PartPooling(y_, with_max_pool=self.with_max_pool)   # [n*s, c]
        
        z = torch.cat([x_, y_], dim=1)     # [n*s, 2*c]
        z = self.fc(z).view(n, s, 2*c, 1, 1).transpose(1, 2)  # [n, 2*c, s, 1, 1]
        
        if self.choosed_part == 'up':
            return x[:,:,:,:h_split,:] * z[:,:c,:,:,:] + y[:,:,:,:h_split,:] * z[:,c:,:,:,:]
        elif self.choosed_part == 'middle':
            return x[:,:,:,h_split:3*h_split,:] * z[:,:c,:,:,:] + y[:,:,:,h_split:3*h_split,:] * z[:,c:,:,:,:]
        elif self.choosed_part == 'down':
            return x[:,:,:,-h_split:,:] * z[:,:c,:,:,:] + y[:,:,:,-h_split:,:] * z[:,c:,:,:,:]


class SeparateFCs(nn.Module):
    def __init__(self, parts_num, in_channels, out_channels, norm=False):
        super(SeparateFCs, self).__init__()
        self.p = parts_num
        self.fc_bin = nn.Parameter(
            nn.init.xavier_uniform_(
                torch.zeros(parts_num, in_channels, out_channels)))
        self.norm = norm

    def forward(self, x):
        """
            x: [n, c_in, p]
            out: [n, c_out, p]
        """
        x = x.permute(2, 0, 1).contiguous()
        if self.norm:
            out = x.matmul(F.normalize(self.fc_bin, dim=1))
        else:
            out = x.matmul(self.fc_bin)
        return out.permute(1, 2, 0).contiguous()


class SeparateBNNecks(nn.Module):
    """
        Bag of Tricks and a Strong Baseline for Deep Person Re-Identification
        CVPR Workshop:  https://openaccess.thecvf.com/content_CVPRW_2019/papers/TRMTMCT/Luo_Bag_of_Tricks_and_a_Strong_Baseline_for_Deep_Person_CVPRW_2019_paper.pdf
        Github: https://github.com/michuanhaohao/reid-strong-baseline
    """

    def __init__(self, parts_num, in_channels, class_num, norm=True, parallel_BN1d=True):
        super(SeparateBNNecks, self).__init__()
        self.p = parts_num
        self.class_num = class_num
        self.norm = norm
        self.fc_bin = nn.Parameter(
            nn.init.xavier_uniform_(
                torch.zeros(parts_num, in_channels, class_num)))
        if parallel_BN1d:
            self.bn1d = nn.BatchNorm1d(in_channels * parts_num)
        else:
            self.bn1d = clones(nn.BatchNorm1d(in_channels), parts_num)
        self.parallel_BN1d = parallel_BN1d

    def forward(self, x):
        """
            x: [n, c, p]
        """
        if self.parallel_BN1d:
            n, c, p = x.size()
            x = x.view(n, -1)  # [n, c*p]
            x = self.bn1d(x)
            x = x.view(n, c, p)
        else:
            x = torch.cat([bn(_x) for _x, bn in zip(
                x.split(1, 2), self.bn1d)], 2)  # [p, n, c]
        feature = x.permute(2, 0, 1).contiguous()
        if self.norm:
            feature = F.normalize(feature, dim=-1)  # [p, n, c]
            logits = feature.matmul(F.normalize(
                self.fc_bin, dim=1))  # [p, n, c]
        else:
            logits = feature.matmul(self.fc_bin)
        return feature.permute(1, 2, 0).contiguous(), logits.permute(1, 2, 0).contiguous()


class FocalConv2d(nn.Module):
    """
        GaitPart: Temporal Part-based Model for Gait Recognition
        CVPR2020: https://openaccess.thecvf.com/content_CVPR_2020/papers/Fan_GaitPart_Temporal_Part-Based_Model_for_Gait_Recognition_CVPR_2020_paper.pdf
        Github: https://github.com/ChaoFan96/GaitPart
    """
    def __init__(self, in_channels, out_channels, kernel_size, halving, **kwargs):
        super(FocalConv2d, self).__init__()
        self.halving = halving
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, bias=False, **kwargs)

    def forward(self, x):
        if self.halving == 0:
            z = self.conv(x)
        else:
            h = x.size(2)
            split_size = int(h // 2**self.halving)
            z = x.split(split_size, 2)
            z = torch.cat([self.conv(_) for _ in z], 2)
        return z


class BasicConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False, **kwargs):
        super(BasicConv3d, self).__init__()
        self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size,
                                stride=stride, padding=padding, bias=bias, **kwargs)

    def forward(self, ipts):
        '''
            ipts: [n, c, s, h, w]
            outs: [n, c, s, h, w]
        '''
        outs = self.conv3d(ipts)
        return outs


class GaitAlign(nn.Module):
    """
        GaitEdge: Beyond Plain End-to-end Gait Recognition for Better Practicality
        ECCV2022: https://arxiv.org/pdf/2203.03972v2.pdf
        Github: https://github.com/ShiqiYu/OpenGait/tree/master/configs/gaitedge
    """
    def __init__(self, H=64, W=44, eps=1, **kwargs):
        super(GaitAlign, self).__init__()
        self.H, self.W, self.eps = H, W, eps
        self.Pad = nn.ZeroPad2d((int(self.W / 2), int(self.W / 2), 0, 0))
        self.RoiPool = RoIAlign((self.H, self.W), 1, sampling_ratio=-1)

    def forward(self, feature_map, binary_mask, w_h_ratio):
        """
           In  sils:         [n, c, h, w]
               w_h_ratio:    [n, 1]
           Out aligned_sils: [n, c, H, W]
        """
        n, c, h, w = feature_map.size()
        # w_h_ratio = w_h_ratio.repeat(1, 1) # [n, 1]
        w_h_ratio = w_h_ratio.view(-1, 1)  # [n, 1]

        h_sum = binary_mask.sum(-1)  # [n, c, h]
        _ = (h_sum >= self.eps).float().cumsum(axis=-1)  # [n, c, h]
        h_top = (_ == 0).float().sum(-1)  # [n, c]
        h_bot = (_ != torch.max(_, dim=-1, keepdim=True)
                 [0]).float().sum(-1) + 1.  # [n, c]

        w_sum = binary_mask.sum(-2)  # [n, c, w]
        w_cumsum = w_sum.cumsum(axis=-1)  # [n, c, w]
        w_h_sum = w_sum.sum(-1).unsqueeze(-1)  # [n, c, 1]
        w_center = (w_cumsum < w_h_sum / 2.).float().sum(-1)  # [n, c]

        p1 = self.W - self.H * w_h_ratio
        p1 = p1 / 2.
        p1 = torch.clamp(p1, min=0)  # [n, c]
        t_w = w_h_ratio * self.H / w
        p2 = p1 / t_w  # [n, c]

        height = h_bot - h_top  # [n, c]
        width = height * w / h  # [n, c]
        width_p = int(self.W / 2)

        feature_map = self.Pad(feature_map)
        w_center = w_center + width_p  # [n, c]

        w_left = w_center - width / 2 - p2  # [n, c]
        w_right = w_center + width / 2 + p2  # [n, c]

        w_left = torch.clamp(w_left, min=0., max=w+2*width_p)
        w_right = torch.clamp(w_right, min=0., max=w+2*width_p)

        boxes = torch.cat([w_left, h_top, w_right, h_bot], dim=-1)
        # index of bbox in batch
        box_index = torch.arange(n, device=feature_map.device)
        rois = torch.cat([box_index.view(-1, 1), boxes], -1)
        crops = self.RoiPool(feature_map, rois)  # [n, c, H, W]
        return crops


def RmBN2dAffine(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight.requires_grad = False
            m.bias.requires_grad = False
