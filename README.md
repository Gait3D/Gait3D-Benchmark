# Gait3D-Benchmark
This repository contains the code and model for our CVPR 2022, ACM MM 2022, 2023, and 2024 papers. 
The Gait3D-Benchmark project is now maintained By [Jinkai Zheng](http://jinkaizheng.com/) and [Xinchen Liu](http://xinchenliu.com/).
Thanks to all of our co-authors for their help, as well as the great repository that we list in the Acknowledgement.

| <h2 align="center"> Gait3D (SMPLGait) </h2> | <h2 align="center"> MTSGait </h2> | <h2 align="center"> Gait3D-Parsing (ParsingGait) </h2> | <h2 align="center"> XGait </h2> |
| :---: | :---: | :---: | :---: |
| Gait Recognition in the Wild with Dense 3D Representations and A Benchmark (CVPR 2022) | Gait Recognition in the Wild with Multi-hop Temporal Switch (ACM MM 2022) | Parsing is All You Need for Accurate Gait Recognition in the Wild (ACM MM 2023) | It Takes Two: Accurate Gait Recognition in the Wild via Cross-granularity Alignment (ACM MM 2024) |
| **[[Project Page]](https://gait3d.github.io) [[Paper]](https://arxiv.org/abs/2204.02569)** | **[[Paper]](https://arxiv.org/abs/2209.00355)** |  **[[Project Page]](https://gait3d.github.io/gait3d-parsing-hp/) [[Paper]](https://arxiv.org/abs/2308.16739)** | **[[Paper]](https://arxiv.org/abs/2411.10742)** |

## What's New
 - [Dec 2024] Our [XGait](https://arxiv.org/abs/2411.10742) method is released.
 - [July 2024] The ACM MM'24 [Multimodal Gait Recognition (MGR) Challenge](https://hcma2024.github.io/mgr) is organized. You can get started quickly [here](https://github.com/Gait3D/Gait3D-Benchmark/tree/main/datasets/MGR).
 - [Sept 2023] The code and model of CDGNet-Parsing are released [here](https://github.com/Gait3D/CDGNet-Parsing), you can use it to extract parsing data on your own data.
 - [Sept 2023] Our [Gait3D-Parsing](https://gait3d.github.io/gait3d-parsing-hp/) dataset and [ParsingGait](https://gait3d.github.io/gait3d-parsing-hp/) method are released.
 - [Sept 2022] Our [MTSGait](https://arxiv.org/abs/2209.00355) method is released.
 - [Mar 2022] Our [Gait3D](https://gait3d.github.io) dataset and [SMPLGait](https://gait3d.github.io) method are released.

## Model Zoo
Results and models are available in the [model zoo](docs/model_zoo.md).

## Requirement and Installation
The requirement and installation procedure can be found [here](docs/requirements.md).

## Data Downloading
Please download the **Gait3D dataset** by signing this [agreement](https://gait3d.github.io/resources/AgreementForGait3D.pdf). 

Please download the **Gait3D-Parsing dataset** by signing this [agreement](https://gait3d.github.io/gait3d-parsing-hp/resources/AgreementForGait3D-Parsing.pdf). 

We ask for your information only to make sure the dataset is used for non-commercial purposes. We will not give it to any third party or publish it publicly anywhere.

### Data Pretreatment
The data pretreatment can be found [here](docs/pretreatment.md).

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

```BibTeX
@inproceedings{zheng2022gait3d,
  title={Gait Recognition in the Wild with Dense 3D Representations and A Benchmark},
  author={Jinkai Zheng, Xinchen Liu, Wu Liu, Lingxiao He, Chenggang Yan, Tao Mei},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2022}
}

@inproceedings{zheng2022mtsgait,
  title={Gait Recognition in the Wild with Multi-hop Temporal Switch},
  author={Jinkai Zheng, Xinchen Liu, Xiaoyan Gu, Yaoqi Sun, Chuang Gan, Jiyong Zhang, Wu Liu, Chenggang Yan},
  booktitle={ACM International Conference on Multimedia (ACM MM)},
  year={2022}
}

@inproceedings{zheng2023parsinggait,
  title={Parsing is All You Need for Accurate Gait Recognition in the Wild},
  author={Jinkai Zheng, Xinchen Liu, Shuai Wang, Lihao Wang, Chenggang Yan, Wu Liu},
  booktitle={ACM International Conference on Multimedia (ACM MM)},
  year={2023}
}

@inproceedings{zheng2024xgait,
  title={It Takes Two: Accurate Gait Recognition in the Wild via Cross-granularity Alignment},
  author={Jinkai Zheng, Xinchen Liu, Boyue Zhang, Chenggang Yan, Jiyong Zhang, Wu Liu, Yongdong Zhang},
  booktitle={ACM International Conference on Multimedia (ACM MM)},
  year={2024}
}
```

## Acknowledgement
Here are some great resources we benefit from:

- The codebase is based on [OpenGait](https://github.com/ShiqiYu/OpenGait).
- The 3D SMPL data is obtained by [ROMP](https://github.com/Arthur151/ROMP).
- The 2D Silhouette data is obtained by [HRNet-segmentation](https://github.com/HRNet/HRNet-Semantic-Segmentation).
- The 2D Parsing data is obtained by [CDGNet](https://github.com/tjpulkl/CDGNet).
- The 2D pose data is obtained by [HRNet](https://github.com/HRNet/HRNet-Human-Pose-Estimation).
- The ReID featrue used to make Gait3D is obtained by [FastReID](https://github.com/JDAI-CV/fast-reid).
