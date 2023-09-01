## Model Zoo
### Gait3D-Parsing
#### Input Size: 64x44

| Method | Rank@1 | Rank@5 | mAP | mINP | download |
| :---: | :---: | :---: |:---: | :---: | :---: |
| [GaitSet(AAAI2019))](https://arxiv.org/pdf/1811.06186.pdf) | 55.90 | 75.60 | 46.69 | 25.61 | [model](https://github.com/Gait3D/Gait3D-Benchmark/releases/download/v1.0/Gait3D-Parsing-GaitSet-180000.pt) |
| [GaitPart(CVPR2020)](http://home.ustc.edu.cn/~saihui/papers/cvpr2020_gaitpart.pdf) | 43.00 | 63.20 | 33.91 | 18.31 | [model](https://github.com/Gait3D/Gait3D-Benchmark/releases/download/v1.0/Gait3D-Parsing-GaitPart-180000.pt) |
| [GLN(ECCV2020)](http://home.ustc.edu.cn/~saihui/papers/eccv2020_gln.pdf) | 45.70 | 67.10 | 38.56 | 20.13 | [model](https://github.com/Gait3D/Gait3D-Benchmark/releases/download/v1.0/Gait3D-Parsing-GLN_P2-180000.pt) |
| [GaitGL(ICCV2021)](https://openaccess.thecvf.com/content/ICCV2021/papers/Lin_Gait_Recognition_via_Effective_Global-Local_Feature_Representation_and_Local_Temporal_ICCV_2021_paper.pdf) | 47.70| 67.20 | 36.23 | 19.35 | [model](https://github.com/Gait3D/Gait3D-Benchmark/releases/download/v1.0/Gait3D-Parsing-GaitGL-180000.pt) |
| [SMPLGait(CVPR2022)](https://gait3d.github.io) | 60.60 | 80.10 | 52.29 | 29.61 | [model](https://github.com/Gait3D/Gait3D-Benchmark/releases/download/v1.0/Gait3D-Parsing-SMPLGait-180000.pt) |
| [MTSGait(ACMMM2023)](https://arxiv.org/abs/2209.00355) | 61.20 | 78.60 | 52.81 | 29.53 | [model](https://github.com/Gait3D/Gait3D-Benchmark/releases/download/v1.0/Gait3D-Parsing-MTSGait-180000.pt) |
| [GaitBase-vanilla(CVPR2023)](https://openaccess.thecvf.com/content/CVPR2023/papers/Fan_OpenGait_Revisiting_Gait_Recognition_Towards_Better_Practicality_CVPR_2023_paper.pdf) | 71.20 | 87.30 | 64.08 | 38.11 | [model](https://github.com/Gait3D/Gait3D-Benchmark/releases/download/v1.0/Gait3D-Parsing-GaitBase_btz32x2_fixed-120000.pt) |
| [ParsingGait(ACMMM2023)](https://arxiv.org/abs/2308.16739) | 76.20 | 89.10 | 68.15 | 41.32 | [model](https://github.com/Gait3D/Gait3D-Benchmark/releases/download/v1.0/Gait3D-Parsing-ParsingGait-120000.pt) |

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
| [MTSGait(ACMMM2023)](https://arxiv.org/abs/2209.00355) | --(48.70) | --(67.10) | --(37.63) | --(21.92) | [model-64](https://github.com/Gait3D/Gait3D-Benchmark/releases/download/v1.0/Gait3D-MTSGait-180000.pt) |

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
    <td class="tg-c3ow">GREW (<a href="https://github.com/Gait3D/Gait3D-Benchmark/blob/main/datasets/GREW/GREW_our_split.json">our split</a>)</td>
    <td class="tg-c3ow">16.50   </td>
    <td class="tg-c3ow">31.10   </td>
    <td class="tg-c3ow">11.71   </td>
  </tr>
  <tr>
    <td class="tg-c3ow" rowspan="2">Gait3D</td>
    <td class="tg-c3ow">GREW (official split)</td>
    <td class="tg-c3ow">18.81   </td>
    <td class="tg-c3ow">32.25   </td>
    <td class="tg-c3ow">~</td>
  </tr>
  <tr>
    <td class="tg-c3ow">GREW (<a href="https://github.com/Gait3D/Gait3D-Benchmark/blob/main/datasets/GREW/GREW_our_split.json">our split</a>)</td>
    <td class="tg-c3ow">43.86   </td>
    <td class="tg-c3ow">60.89   </td>
    <td class="tg-c3ow">28.06   </td>
  </tr>
</tbody>
</table>
