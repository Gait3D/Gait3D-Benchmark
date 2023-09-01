# # **************** For Gait3D ****************
# # Baseline
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 opengait/main.py --cfgs ./configs/baseline/baseline_gait3d.yaml --phase train

# # GaitBase
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 opengait/main.py --cfgs ./configs/gaitbase/gaitbase_gait3d_btz32x2_fixed.yaml --phase train

# # GaitSet
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 opengait/main.py --cfgs ./configs/gaitset/gaitset_gait3d.yaml --phase train

# # GaitPart
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 opengait/main.py --cfgs ./configs/gaitpart/gaitpart_gait3d.yaml --phase train

# GaitGL
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --master_port 12345 --nproc_per_node=4 opengait/main.py --cfgs ./configs/gaitgl/gaitgl_gait3d.yaml --phase train

# # GLN 
# # Phase 1
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --master_port 12345  --nproc_per_node=2 opengait/main.py --cfgs ./configs/gln/gln_phase1_gait3d.yaml --phase train
# # Phase 2
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs ./configs/gln/gln_phase2_gait3d.yaml --phase train

# SMPLGait
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --master_port 12345 --nproc_per_node=4 opengait/main.py --cfgs ./configs/smplgait/smplgait_gait3d.yaml --phase train

# MTSGait
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --master_port 12345 --nproc_per_node=4 opengait/main.py --cfgs ./configs/mtsgait/mtsgait_gait3d.yaml --phase train



# # **************** For Gait3D-Parsing ****************
# # GaitBase
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 opengait/main.py --cfgs ./configs/gaitbase/gaitbase_gait3d_parsing_btz32x2_fixed.yaml --phase train

# # GaitSet
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 opengait/main.py --cfgs ./configs/gaitset/gaitset_gait3d_parsing.yaml --phase train

# # GaitPart
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 opengait/main.py --cfgs ./configs/gaitpart/gaitpart_gait3d_parsing.yaml --phase train

# GaitGL
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --master_port 12345 --nproc_per_node=4 opengait/main.py --cfgs ./configs/gaitgl/gaitgl_gait3d_parsing.yaml --phase train

# # GLN 
# # Phase 1
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --master_port 12345  --nproc_per_node=2 opengait/main.py --cfgs ./configs/gln/gln_phase1_gait3d_parsing.yaml --phase train
# # Phase 2
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs ./configs/gln/gln_phase2_gait3d_parsing.yaml --phase train

# SMPLGait
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --master_port 12345 --nproc_per_node=4 opengait/main.py --cfgs ./configs/smplgait/smplgait_gait3d_parsing.yaml --phase train

# MTSGait
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --master_port 12345 --nproc_per_node=4 opengait/main.py --cfgs ./configs/mtsgait/mtsgait_gait3d_parsing.yaml --phase train

# ParsingGait
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --master_port 12345 --nproc_per_node=4 opengait/main.py --cfgs ./configs/parsinggait/parsinggait_gait3d_parsing.yaml --phase train
