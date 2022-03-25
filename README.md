# Gait3D-Benchmark
This is the code for the paper "Jinkai Zheng, Xinchen Liu, Wu Liu, Lingxiao He, Chenggang Yan, Tao Mei: [Gait Recognition in the Wild with Dense 3D Representations and A Benchmark](https://gait3d.github.io). (CVPR 2022)"


## What's New
 - [Mar 2022] Another gait in the wild dataset [GREW](https://www.grew-benchmark.org/) is supported.
 - [Mar 2022] Our [Gait3D](https://gait3d.github.io) dataset and [SMPLGait](https://gait3d.github.io) method are released.


## Model Zoo
### Gait3D
#### Input Size: 128x88(64x44)

| Method | Rank@1 | Rank@5 | mAP | mINP | download |
| :---: | :---: | :---: |:---: | :---: | :---: |
| [GaitSet(AAAI2019))](https://arxiv.org/pdf/1811.06186.pdf) | 42.60(36.70) | 63.10(58.30) | 33.69(30.01) | 19.69(17.30) | [model-128](https://github.com/Gait3D/Gait3D-Benchmark/releases/download/v0.1/128pixel-GaitSet-180000.pt)([model-64](https://github.com/Gait3D/Gait3D-Benchmark/releases/download/v0.1/64pixel-GaitSet-180000.pt)) |
| [GaitPart(CVPR2020)](http://home.ustc.edu.cn/~saihui/papers/cvpr2020_gaitpart.pdf) | 29.90(28.20) | 50.60(47.60) | 23.34(21.58) | 13.15(12.36) | [model-128](https://github.com/Gait3D/Gait3D-Benchmark/releases/download/v0.1/128pixel-GaitPart-180000.pt)([model-64](https://github.com/Gait3D/Gait3D-Benchmark/releases/download/v0.1/64pixel-GaitPart-180000.pt)) |
| [GLN(ECCV2020)](http://home.ustc.edu.cn/~saihui/papers/eccv2020_gln.pdf) | 42.20(31.40) | 64.50(52.90) | 33.14(24.74) | 19.56(13.58) | [model-128](https://github.com/Gait3D/Gait3D-Benchmark/releases/download/v0.1/128pixel-GLN_P2-180000.pt)([model-64](https://github.com/Gait3D/Gait3D-Benchmark/releases/download/v0.1/64pixel-GLN_P2-180000.pt)) |
| [GaitGL(ICCV2021)](https://openaccess.thecvf.com/content/ICCV2021/papers/Lin_Gait_Recognition_via_Effective_Global-Local_Feature_Representation_and_Local_Temporal_ICCV_2021_paper.pdf) | 23.50(29.70)| 38.50(48.50) | 16.40(22.29) | 9.20(13.26) | [model-128](https://github.com/Gait3D/Gait3D-Benchmark/releases/download/v0.1/128pixel-GaitGL-180000.pt)([model-64](https://github.com/Gait3D/Gait3D-Benchmark/releases/download/v0.1/64pixel-GaitGL-180000.pt)) |
| [OpenGait Baseline*](https://github.com/ShiqiYu/OpenGait) | 47.70(42.90) | 67.20(63.90) | 37.62(35.19) | 22.24(20.83) | [model-128](https://github.com/Gait3D/Gait3D-Benchmark/releases/download/v0.1/128pixel-Baseline-180000.pt)([model-64](https://github.com/Gait3D/Gait3D-Benchmark/releases/download/v0.1/64pixel-Baseline-180000.pt)) |
| [SMPLGait(CVPR2022)](https://gait3d.github.io) | 53.20(46.30) | 71.00(64.50) | 42.43(37.16) | 25.97(22.23) | [model-128](https://github.com/Gait3D/Gait3D-Benchmark/releases/download/v0.1/128pixel-SMPLGait_128pixel-180000.pt)([model-64](https://github.com/Gait3D/Gait3D-Benchmark/releases/download/v0.1/64pixel-SMPLGait_64pixel-180000.pt)) |

*It should be noticed that OpenGait Baseline is equal to SMPLGait w/o 3D in our paper.

### Cross Domain 
#### Datasets in the Wild (GaitSet, 64x44)

<table class="tg">
<thead>
  <tr>
    <th class="tg-c3ow">Source</th>
    <th class="tg-c3ow">Target</th>
    <th class="tg-c3ow">Rank@1</th>
    <th class="tg-c3ow">Rank@5</th>
    <th class="tg-c3ow">mAP</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-c3ow">GREW (official split)</td>
    <td class="tg-c3ow" rowspan="2">Gait3D</td>
    <td class="tg-c3ow">15.80   </td>
    <td class="tg-c3ow">30.20   </td>
    <td class="tg-c3ow">11.83   </td>
  </tr>
  <tr>
    <td class="tg-c3ow">GREW (our split)</td>
    <td class="tg-c3ow">16.50   </td>
    <td class="tg-c3ow">31.10   </td>
    <td class="tg-c3ow">11.71   </td>
  </tr>
  <tr>
    <td class="tg-c3ow" rowspan="2">Gait3D</td>
    <td class="tg-c3ow">GREW   (official split)</td>
    <td class="tg-c3ow">18.81   </td>
    <td class="tg-c3ow">32.25   </td>
    <td class="tg-c3ow">~</td>
  </tr>
  <tr>
    <td class="tg-c3ow">GREW (our split)</td>
    <td class="tg-c3ow">43.86   </td>
    <td class="tg-c3ow">60.89   </td>
    <td class="tg-c3ow">28.06   </td>
  </tr>
</tbody>
</table>


## Requirements
- pytorch >= 1.6
- torchvision
- pyyaml
- tensorboard
- opencv-python
- tqdm
- py7zr
- tabulate
- termcolor

### Installation
You can replace the second command from the bottom to install
[pytorch](https://pytorch.org/get-started/previous-versions/#v110) 
based on your CUDA version.
```
git clone https://github.com/Gait3D/Gait3D-Benchmark.git
cd Gait3D-Benchmark
conda create --name py37torch160 python=3.7
conda activate py37torch160
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.2 -c pytorch
pip install tqdm pyyaml tensorboard opencv-python tqdm py7zr tabulate termcolor
```


## Data Preparation
Download [Gait3D](https://gait3d.github.io/resources/AgreementForGait3D.pdf) dataset.

### Data Pretreatment
Run the following command to preprocess the Gait3D dataset.
```
python misc/pretreatment.py --input_path 'Gait3D/2D_Silhouettes' --output_path 'Gait3D-sils-64-44-pkl' --img_h 64 --img_w 44
python misc/pretreatment.py --input_path 'Gait3D/2D_Silhouettes' --output_path 'Gait3D-sils-128-88-pkl' --img_h 128 --img_w 88
python misc/pretreatment_smpl.py --input_path 'Gait3D/3D_SMPLs' --output_path 'Gait3D-smpls-pkl'
```

### Data Structrue
After the pretreatment, the data structure under the directory should like this
```
├── Gait3D-sils-64-44-pkl
│  ├── 0000
│     ├── camid0_videoid2
│        ├── seq0
│           └──seq0.pkl
├── Gait3D-sils-128-88-pkl
│  ├── 0000
│     ├── camid0_videoid2
│        ├── seq0
│           └──seq0.pkl
├── Gait3D-smpls-pkl
│  ├── 0000
│     ├── camid0_videoid2
│        ├── seq0
│           └──seq0.pkl
```


## Train
Run the following command:
```bash
sh train.sh
```

## Test
Run the following command:
```bash
sh test.sh
```



## Citation
Please cite this paper in your publications if it helps your research:
```
@inproceedings{zheng2022gait3d,
title={Gait Recognition in the Wild with Dense 3D Representations and A Benchmark},
author={Jinkai Zheng, Xinchen Liu, Wu Liu, Lingxiao He, Chenggang Yan, Tao Mei},
booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
year={2022}
}
```


## Acknowledgement
- [OpenGait](https://github.com/ShiqiYu/OpenGait)
