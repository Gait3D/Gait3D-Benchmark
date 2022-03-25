import os
import pickle
import os.path as osp
import torch.utils.data as tordata
import json
from utils import get_msg_mgr
from tabulate import tabulate
from termcolor import colored


class DataSet_SMPLGait(tordata.Dataset):
    def __init__(self, data_cfg, training):
        """
            seqs_info: the list with each element indicating
                            a certain gait sequence presented as [label, type, view, paths];
        """
        self.__dataset_parser(data_cfg, training)
        self.cache = data_cfg['cache']
        self.label_list = [seq_info[0] for seq_info in self.seqs_info]
        self.types_list = [seq_info[1] for seq_info in self.seqs_info]
        self.views_list = [seq_info[2] for seq_info in self.seqs_info]

        self.label_set = sorted(list(set(self.label_list)))
        self.types_set = sorted(list(set(self.types_list)))
        self.views_set = sorted(list(set(self.views_list)))
        self.seqs_data = [None] * len(self)
        self.indices_dict = {label: [] for label in self.label_set}
        for i, seq_info in enumerate(self.seqs_info):
            self.indices_dict[seq_info[0]].append(i)
        if self.cache:
            self.__load_all_data()

    def __len__(self):
        return len(self.seqs_info)

    def __loader__(self, paths):
        paths = sorted(paths)
        data_list = []
        for pths in paths:
            sil_pth = pths[0]
            if sil_pth.endswith('.pkl'):
                with open(sil_pth, 'rb') as f:
                    sil_data = pickle.load(f)
                f.close()
            sp_pth = pths[1]
            if sp_pth.endswith('.pkl'):
                with open(sp_pth, 'rb') as f:
                    sp_data = pickle.load(f)
                f.close()
            else:
                raise ValueError('- Loader - just support .pkl !!!')

            data_list.append([sil_data, sp_data])
        for idx, data in enumerate(data_list):
            if len(data[0]) != len(data_list[0][0]):
                raise ValueError(
                    'Each input sil data({}) should have the same length.'.format(paths[idx][0]))
            if len(data[1]) != len(data_list[0][1]):
                raise ValueError(
                    'Each input smpl data({}) should have the same length.'.format(paths[idx][1]))
            if len(data[0]) != len(data[1]):
                raise ValueError(
                    'Each input sil data({}) should have the same length to smpl data({}).'
                        .format(paths[idx][0], paths[idx][1]))
            if len(data[0]) == 0 or len(data[1]) == 0:
                raise ValueError(
                    'Each input sil data({}) and smpl data({}) should have at least one element.'
                        .format(paths[idx][0], paths[idx][1]))

        return data_list

    def __getitem__(self, idx):
        if not self.cache:
            data_list = self.__loader__(self.seqs_info[idx][-1])
        elif self.seqs_data[idx] is None:
            data_list = self.__loader__(self.seqs_info[idx][-1])
            self.seqs_data[idx] = data_list
        else:
            data_list = self.seqs_data[idx]
        seq_info = self.seqs_info[idx]
        return data_list, seq_info

    def __load_all_data(self):
        for idx in range(len(self)):
            self.__getitem__(idx)

    def __load_seqs_into_list(self, dataset_root, smpl_root, lab, cam, seq, seqs_info_list, data_in_use):
        msg_mgr = get_msg_mgr()
        seq_paras = [lab, cam, seq]
        seq_path = osp.join(dataset_root, *seq_paras)
        smpl_path = osp.join(smpl_root, *seq_paras)

        seq_dirs = sorted(os.listdir(seq_path))

        cam_typ = cam.split("_videoid")[0]
        cam_id = int(cam_typ.split("camid")[1])
        seq_info = [lab, cam_id, seq]

        if seq_dirs != []:
            seq_dirs = [[osp.join(seq_path, dir), osp.join(smpl_path, dir)]
                        for dir in seq_dirs]
            if data_in_use is not None:
                seq_dirs = [dir for dir, use_bl in zip(
                    seq_dirs, data_in_use) if use_bl]
            seqs_info_list.append([*seq_info, seq_dirs])
        else:
            msg_mgr.log_debug('Find no .pkl file in %s-%s-%s.' % (lab, cam, seq))

    def __print_dataset_csv_format(self, dataset, train_info, probe_info, gallery_info):
        """
        Print main metrics in a format similar to Detectron,
        so that they are easy to copypaste into a spreadsheet.
        Args:
            results (OrderedDict[dict]): task_name -> {metric -> score}
        """
        msg_mgr = get_msg_mgr()
        if train_info != {}:
            headers = list(train_info.keys())
            csv_results = []
            for data_info in [train_info]:
                csv_results.append((data_info.values()))
        else:
            headers = list(probe_info.keys())
            csv_results = []
            for data_info in [probe_info, gallery_info]:
                csv_results.append((data_info.values()))
        # tabulate it
        table = tabulate(
            csv_results,
            tablefmt="psql",
            headers=headers,
            numalign="left",
        )
        msg_mgr.log_info(f"Load {dataset} in csv format: \n" + colored(table, "cyan"))

    def __visualize_data_info(self, dataset_name, label_set, probe_seqs_info_list, seqs_info_list):
        if probe_seqs_info_list != []:
            probe_info = {
                "subset": "probe",
                "ids": len(set(label_set)),
                "seqs": len(probe_seqs_info_list),
            }
            gallery_info = {
                "subset": "gallery",
                "ids": len(set(label_set)),
                "seqs": len(seqs_info_list),
            }
            self.__print_dataset_csv_format(f"{dataset_name}-testset", train_info={}, probe_info=probe_info,
                                            gallery_info=gallery_info)
        if probe_seqs_info_list == []:
            train_info = {
                "subset": "train",
                "ids": len(set(label_set)),
                "seqs": len(seqs_info_list),
            }
            self.__print_dataset_csv_format(f"{dataset_name}-trainset", train_info=train_info, probe_info={},
                                            gallery_info={})

    def __dataset_parser(self, data_config, training):
        dataset_root = data_config['dataset_root']['silhouette_root']
        smpl_root = data_config['dataset_root']['smpl_root']
        try:
            data_in_use = data_config['data_in_use']  # [n], true or false
        except:
            data_in_use = None

        with open(data_config['dataset_partition'], "rb") as f:
            partition = json.load(f)
        train_set = partition["TRAIN_SET"]
        test_set = partition["TEST_SET"]
        probe_set = partition["PROBE_SET"]
        label_list = os.listdir(dataset_root)
        train_set = [label for label in train_set if label in label_list]
        test_set = [label for label in test_set if label in label_list]
        miss_pids = [label for label in label_list if label not in (
                train_set + test_set)]
        msg_mgr = get_msg_mgr()

        def log_pid_list(pid_list):
            if len(pid_list) >= 3:
                msg_mgr.log_info('[%s, %s, ..., %s]' %
                                 (pid_list[0], pid_list[1], pid_list[-1]))
            else:
                msg_mgr.log_info(pid_list)

        if len(miss_pids) > 0:
            msg_mgr.log_debug('-------- Miss Pid List --------')
            msg_mgr.log_debug(miss_pids)
        if training:
            msg_mgr.log_info("-------- Train Pid List --------")
            log_pid_list(train_set)
        else:
            msg_mgr.log_info("-------- Test Pid List --------")
            log_pid_list(test_set)

        def get_seqs_info_list(label_set):
            if training:
                dataset_name = data_config['dataset_name']
            else:
                dataset_name = data_config['test_dataset_name']
            seqs_info_list = []
            probe_seqs_info_list = []
            for lab in label_set:
                for cam in sorted(os.listdir(osp.join(dataset_root, lab))):
                    for seq in sorted(os.listdir(osp.join(dataset_root, lab, cam))):
                        id_cam_seq = f"{lab}-{cam}-{seq}"
                        if id_cam_seq in probe_set:
                            self.__load_seqs_into_list(dataset_root, smpl_root,
                                                       lab, cam, seq,
                                                       probe_seqs_info_list, data_in_use)
                            continue
                        self.__load_seqs_into_list(dataset_root, smpl_root,
                                                   lab, cam, seq,
                                                   seqs_info_list, data_in_use)
            self.__visualize_data_info(dataset_name, label_set, probe_seqs_info_list, seqs_info_list)

            return probe_seqs_info_list + seqs_info_list, len(probe_seqs_info_list)

        self.seqs_info, self.probe_seqs_num = get_seqs_info_list(
            train_set) if training else get_seqs_info_list(test_set)

