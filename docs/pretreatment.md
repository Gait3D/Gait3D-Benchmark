### Data Pretreatment
Run the following command to preprocess the Gait3D dataset.
```
python datasets/pretreatment.py -i 'Gait3D/2D_Silhouettes' -o 'Gait3D-sils-64-64-pkl' -r 64
python datasets/pretreatment.py -i 'Gait3D/2D_Silhouettes' -o 'Gait3D-sils-128-128-pkl' -r 128
python datasets/Gait3D/pretreatment_smpl.py -i 'Gait3D/3D_SMPLs' -o 'Gait3D-smpls-pkl'
```

Run the following command to preprocess the Gait3D-Parsing dataset.
```
python datasets/pretreatment_gps.py -i 'Gait3D/2D_Parsings' -o 'Gait3D-pars-64-64-pkl' -r 64 -p
```

Run the following command to merge the sils, smpls, and pars dataset.
```
python datasets/Gait3D/merge_three_modality.py --pars_path 'Gait3D-pars-64-64-pkl' --sils_path 'Gait3D-sils-64-64-pkl' --smpls_path 'Gait3D-smpls-pkl' --output_path 'Gait3D-merged-pkl' --link 'hard'
```

### Data Structrue
After the pretreatment, the data structure under the directory should like this
```
├── Gait3D-sils-64-64-pkl
│  ├── 0000
│     ├── camid0_videoid2
│        ├── seq0
│           └──seq0.pkl
├── Gait3D-sils-128-128-pkl
│  ├── 0000
│     ├── camid0_videoid2
│        ├── seq0
│           └──seq0.pkl
├── Gait3D-smpls-pkl
│  ├── 0000
│     ├── camid0_videoid2
│        ├── seq0
│           └──seq0.pkl
├── Gait3D-pars-64-64-pkl
│  ├── 0000
│     ├── camid0_videoid2
│        ├── seq0
│           └──seq0.pkl
├── Gait3D-merged-pkl
│  ├── 0000
│     ├── camid0_videoid2
│        ├── seq0
│           ├──pars-seq0.pkl
│           ├──sils-seq0.pkl
│           └──smpls-seq0.pkl
```