# The ACM MM'24 Multimodal Gait Recognition (MGR) Challenge
This is a support for [Multimodal Gait Recognition (MGR)](https://hcma2024.github.io/mgr) Challenge. We provide the baseline code for this competition.

## Tutorial for MGR 2024
The MGR 2024 challenge is divided into four tracks, a team can participate in no more than two (≤2) of the four race tracks, and only one of Track 1 and Track 2 can be chosen.

For more details, please refer to [here](https://hcma2024.github.io/mgr).

In this project, we take the track 1 (Parsing-based Gait Recognition Track) as an example for demonstration.


### Download the dataset
Download the Trainset, Valset/Testset (gallery and probe) from the [Track 1 Codalab website](https://codalab.lisn.upsaclay.fr/competitions/20040).
You should decompress these files by following command:
```
# For Phase 1
mkdir mgr_2024_track1_phase1
unzip MGR24_TrainSet_Parsing_pkl_1000IDs.zip
mv MGR24_TrainSet_Parsing_pkl_1000IDs/* mgr_2024_track1_phase1/
rm MGR24_TrainSet_Parsing_pkl_1000IDs -rf

unzip MGR24_ValSet_Parsing_Gallery_pkl.zip
mv MGR24_ValSet_Parsing_Gallery_pkl/* mgr_2024_track1_phase1/
rm MGR24_ValSet_Parsing_Gallery_pkl -rf

unzip MGR24_ValSet_Parsing_Probe_pkl.zip
mkdir -P mgr_2024_track1_phase1/probe
mv MGR24_ValSet_Parsing_Probe_pkl/* mgr_2024_track1_phase1/probe/
rm MGR24_ValSet_Parsing_Probe_pkl -rf

# For Phase 2
mkdir mgr_2024_track1_phase2
unzip MGR24_TrainSet_Parsing_pkl_1000IDs.zip
mv MGR24_TrainSet_Parsing_pkl_1000IDs/* mgr_2024_track1_phase2/
rm MGR24_TrainSet_Parsing_pkl_1000IDs -rf

unzip MGR24_TestSet_Parsing_Gallery_pkl.zip
mv MGR24_TestSet_Parsing_Gallery_pkl/* mgr_2024_track1_phase2/
rm MGR24_TestSet_Parsing_Gallery_pkl -rf

unzip MGR24_TestSet_Parsing_Probe_pkl.zip
mkdir -P mgr_2024_track1_phase2/probe
mv MGR24_TestSet_Parsing_Probe_pkl/* mgr_2024_track1_phase2/probe/
rm MGR24_TestSet_Parsing_Probe_pkl -rf
```


### Train the dataset
For the phase 1:

Modify the `dataset_root` in `configs/baseline/baseline_mgr_track1_phase1.yaml`, and then run this command:
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 opengait/main.py --cfgs configs/baseline/baseline_mgr_track1_phase1.yaml --phase train
```

For the phase 2, please replace 'phase1' with 'phase2' in the config file name.


## Generate the result
For the phase 1:

Modify the `dataset_root` in `configs/baseline/baseline_mgr_track1_phase1.yaml`, and then run this command:
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 opengait/main.py --cfgs configs/baseline/baseline_mgr_track1_phase1.yaml --phase test
```
The result will be generated in `MGR_result/current_time.csv`.

For the phase 2, please replace 'phase1' with 'phase2' in the config file name.


## Submit the result
Rename the csv file to `submission.csv`, then zip it and upload to [official Track 1 Codalab submission link](https://codalab.lisn.upsaclay.fr/competitions/20040#participate).
Normally, you should get a score.

---

