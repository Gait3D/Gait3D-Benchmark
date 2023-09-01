## Requirements
- pytorch >= 1.12.1
- torchvision
- pyyaml
- tensorboard
- opencv-python
- tqdm
- kornia

## Installation
You can replace the second command from the bottom to install
[pytorch](https://pytorch.org/get-started/previous-versions/#v110) 
based on your CUDA version.
```
git clone https://github.com/Gait3D/Gait3D-Benchmark.git
cd Gait3D-Benchmark
conda create -n py38torch1121 python=3.8
conda activate py38torch1121
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=10.2 -c pytorch
pip install pyyaml tensorboard opencv-python tqdm kornia
```