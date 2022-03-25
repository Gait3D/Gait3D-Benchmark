import csv
import os
import random

import torch
import numpy as np
import torch.nn.functional as F
from utils import get_msg_mgr
from collections import OrderedDict

from tabulate import tabulate
from termcolor import colored

from utils import get_msg_mgr
from utils import evaluate_rank
from utils import FAMOUS_SAYINGS


def print_csv_format(dataset, results):
    """
    Print main metrics in a format similar to Detectron,
    so that they are easy to copypaste into a spreadsheet.
    Args:
        results (OrderedDict[dict]): task_name -> {metric -> score}
    """
    assert isinstance(results, OrderedDict), results  # unordered results cannot be properly printed
    msg_mgr = get_msg_mgr()
    metrics = ["Dataset"]
    metrics.extend(list(results.keys()))

    csv_results = []
    csv_results.append((dataset, *list(results.values())))

    # tabulate it
    table = tabulate(
        csv_results,
        tablefmt="psql",
        floatfmt=".2f",
        headers=metrics,
        numalign="left",
    )
    msg_mgr.log_info("Evaluation results in csv format: \n" + colored(table, "cyan"))


def cuda_dist(x, y, metric='euc'):
    x = torch.from_numpy(x).cuda()
    y = torch.from_numpy(y).cuda()
    if metric == 'cos':
        x = F.normalize(x, p=2, dim=2)  # n p c
        y = F.normalize(y, p=2, dim=2)  # n p c
    num_bin = x.size(1)
    n_x = x.size(0)
    n_y = y.size(0)
    dist = torch.zeros(n_x, n_y).cuda()
    for i in range(num_bin):
        _x = x[:, i, ...]
        _y = y[:, i, ...]
        if metric == 'cos':
            dist += torch.matmul(_x, _y.transpose(0, 1))
        else:
            _dist = torch.sum(_x ** 2, 1).unsqueeze(1) + torch.sum(_y ** 2, 1).unsqueeze(
                1).transpose(0, 1) - 2 * torch.matmul(_x, _y.transpose(0, 1))
            dist += torch.sqrt(F.relu(_dist))
    return 1 - dist/num_bin if metric == 'cos' else dist / num_bin

# Exclude identical-view cases


def de_diag(acc, each_angle=False):
    dividend = acc.shape[1] - 1.
    result = np.sum(acc - np.diag(np.diag(acc)), 1) / dividend
    if not each_angle:
        result = np.mean(result)
    return result

# Modified From https://github.com/AbnerHqC/GaitSet/blob/master/model/utils/evaluator.py


def identification(data, dataset, metric='euc'):
    msg_mgr = get_msg_mgr()

    feature, label, seq_type, view = data['embeddings'], data['labels'], data['types'], data['views']
    label = np.array(label)
    view_list = list(set(view))
    view_list.sort()
    view_num = len(view_list)
    # sample_num = len(feature)

    probe_seq_dict = {'CASIA-B': [['nm-05', 'nm-06'], ['bg-01', 'bg-02'], ['cl-01', 'cl-02']],
                      'OUMVLP': [['00']]}

    gallery_seq_dict = {'CASIA-B': [['nm-01', 'nm-02', 'nm-03', 'nm-04']],
                        'OUMVLP': [['01']]}
    if dataset not in (probe_seq_dict or gallery_seq_dict):
        raise KeyError("DataSet %s hasn't been supported !" % dataset)
    num_rank = 5
    acc = np.zeros([len(probe_seq_dict[dataset]),
                    view_num, view_num, num_rank]) - 1.
    for (p, probe_seq) in enumerate(probe_seq_dict[dataset]):
        for gallery_seq in gallery_seq_dict[dataset]:
            for (v1, probe_view) in enumerate(view_list):
                for (v2, gallery_view) in enumerate(view_list):
                    gseq_mask = np.isin(seq_type, gallery_seq) & np.isin(
                        view, [gallery_view])
                    gallery_x = feature[gseq_mask, :]
                    gallery_y = label[gseq_mask]

                    pseq_mask = np.isin(seq_type, probe_seq) & np.isin(
                        view, [probe_view])
                    probe_x = feature[pseq_mask, :]
                    probe_y = label[pseq_mask]

                    dist = cuda_dist(probe_x, gallery_x, metric)
                    idx = dist.sort(1)[1].cpu().numpy()
                    acc[p, v1, v2, :] = np.round(
                        np.sum(np.cumsum(np.reshape(probe_y, [-1, 1]) == gallery_y[idx[:, 0:num_rank]], 1) > 0,
                               0) * 100 / dist.shape[0], 2)
    result_dict = {}
    if 'OUMVLP' not in dataset:
        for i in range(1):
            msg_mgr.log_info(
                '===Rank-%d (Include identical-view cases)===' % (i + 1))
            msg_mgr.log_info('NM: %.3f,\tBG: %.3f,\tCL: %.3f' % (
                np.mean(acc[0, :, :, i]),
                np.mean(acc[1, :, :, i]),
                np.mean(acc[2, :, :, i])))
        for i in range(1):
            msg_mgr.log_info(
                '===Rank-%d (Exclude identical-view cases)===' % (i + 1))
            msg_mgr.log_info('NM: %.3f,\tBG: %.3f,\tCL: %.3f' % (
                de_diag(acc[0, :, :, i]),
                de_diag(acc[1, :, :, i]),
                de_diag(acc[2, :, :, i])))
        result_dict["scalar/test_accuracy/NM"] = de_diag(acc[0, :, :, i])
        result_dict["scalar/test_accuracy/BG"] = de_diag(acc[1, :, :, i])
        result_dict["scalar/test_accuracy/CL"] = de_diag(acc[2, :, :, i])
        np.set_printoptions(precision=2, floatmode='fixed')
        for i in range(1):
            msg_mgr.log_info(
                '===Rank-%d of each angle (Exclude identical-view cases)===' % (i + 1))
            msg_mgr.log_info('NM: {}'.format(de_diag(acc[0, :, :, i], True)))
            msg_mgr.log_info('BG: {}'.format(de_diag(acc[1, :, :, i], True)))
            msg_mgr.log_info('CL: {}'.format(de_diag(acc[2, :, :, i], True)))
    else:
        msg_mgr.log_info('===Rank-1 (Include identical-view cases)===')
        msg_mgr.log_info('NM: %.3f ' % (np.mean(acc[0, :, :, 0])))
        msg_mgr.log_info('===Rank-1 (Exclude identical-view cases)===')
        msg_mgr.log_info('NM: %.3f ' % (de_diag(acc[0, :, :, 0])))
        result_dict["scalar/test_accuracy/NM"] = de_diag(acc[0, :, :, 0])

    print_famous_saying(msg_mgr)

    return result_dict


def identification_real_scene(data, dataset, metric='euc'):
    msg_mgr = get_msg_mgr()
    feature, label, seq_type = data['embeddings'], data['labels'], data['types']
    label = np.array(label)

    gallery_seq_type = {'0001-1000': ['1', '2'],
                        "HID2021": ['0'], '0001-1000-test': ['0']}
    probe_seq_type = {'0001-1000': ['3', '4', '5', '6'],
                      "HID2021": ['1'], '0001-1000-test': ['1']}

    num_rank = 5
    acc = np.zeros([num_rank]) - 1.
    gseq_mask = np.isin(seq_type, gallery_seq_type[dataset])
    gallery_x = feature[gseq_mask, :]
    gallery_y = label[gseq_mask]
    pseq_mask = np.isin(seq_type, probe_seq_type[dataset])
    probe_x = feature[pseq_mask, :]
    probe_y = label[pseq_mask]

    dist = cuda_dist(probe_x, gallery_x, metric)
    idx = dist.cpu().sort(1)[1].numpy()
    acc = np.round(np.sum(np.cumsum(np.reshape(probe_y, [-1, 1]) == gallery_y[idx[:, 0:num_rank]], 1) > 0,
                          0) * 100 / dist.shape[0], 2)
    msg_mgr.log_info('==Rank-1==')
    msg_mgr.log_info('%.3f' % (np.mean(acc[0])))
    msg_mgr.log_info('==Rank-5==')
    msg_mgr.log_info('%.3f' % (np.mean(acc[4])))

    print_famous_saying(msg_mgr)

    return {"scalar/test_accuracy/Rank-1": np.mean(acc[0]), "scalar/test_accuracy/Rank-5": np.mean(acc[4])}


def evaluation_Gait3D(data, conf, probe_num, metric='euc'):
    msg_mgr = get_msg_mgr()

    dataset_name = conf["data_cfg"]["test_dataset_name"]

    features, labels, cams, time_seqs = data['embeddings'], data['labels'], data['types'], data['views']

    probe_features = features[:probe_num]
    gallery_features = features[probe_num:]
    probe_lbls = np.asarray(labels[:probe_num])
    gallery_lbls = np.asarray(labels[probe_num:])

    results = OrderedDict()
    msg_mgr.log_info(f"The test metric you choose is {metric}.")
    dist = cuda_dist(probe_features, gallery_features, metric).cpu().numpy()
    cmc, all_AP, all_INP = evaluate_rank(dist, probe_lbls, gallery_lbls)

    mAP = np.mean(all_AP)
    mINP = np.mean(all_INP)
    for r in [1, 5, 10]:
        results['Rank-{}'.format(r)] = cmc[r - 1] * 100
    results['mAP'] = mAP * 100
    results['mINP'] = mINP * 100

    print_csv_format(dataset_name, results)

    print_famous_saying(msg_mgr)

    return results


def evaluation_GREW(data, conf, probe_num, metric='euc'):
    msg_mgr = get_msg_mgr()
    # read submission.csv
    submission_path = conf["evaluator_cfg"]["submission_path"]
    with open(submission_path, 'r') as f:
        reader = csv.reader(f)
        listcsv = []
        for i, row in enumerate(reader):
            listcsv.append(row)

    rank = 20
    features, labels, cams, time_seqs = data['embeddings'], data['labels'], data['types'], data['views']

    probe_lbls = np.asarray(labels[:probe_num])
    gallery_lbls = np.asarray(labels[probe_num:])

    msg_mgr.log_info(f"The test metric you choose is {metric}.")
    probe_features = features[:probe_num]
    gallery_features = features[probe_num:]

    dist = cuda_dist(probe_features, gallery_features, metric)
    idx = dist.sort(1)[1].cpu().numpy()

    for i, vidId in enumerate(probe_lbls):
        for j, _idx in enumerate(idx[i][:rank]):
            listcsv[i+1][0] = vidId
            listcsv[i+1][j + 1] = int(gallery_lbls[_idx])

    with open(submission_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in listcsv:
            writer.writerow(row)

    msg_mgr.log_info(f"The results have been saved to {submission_path}.")
    msg_mgr.log_info(f"Please zip the CSV file and upload it to the following URL:")
    msg_mgr.log_info(f"https://competitions.codalab.org/competitions/35463")
    print_famous_saying(msg_mgr)


def print_famous_saying(msg_mgr):
    random.seed(None)
    msg_mgr.log_info(f"Duang~Duang~Duang~ Here is a famous saying for you.")
    msg_mgr.log_info(f"\033[1;32m{FAMOUS_SAYINGS[random.randint(0, 49)]}\033[0m")
    msg_mgr.log_info(f"Best Wishes!")
    msg_mgr.log_info(f"-- The Group of Gait3D")
