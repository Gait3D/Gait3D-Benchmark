# # ---------------------Pixel is [64, 44]--------------------- # #
# # GaitSet
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 lib/main.py --cfgs ./config/gaitset_64pixel.yaml --phase test
# # GaitPart
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 lib/main.py --cfgs ./config/gaitpart_64pixel.yaml --phase test
# # GLN
# Phase 1
#CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --master_port 12345  --nproc_per_node=2 lib/main.py --cfgs ./config/gln/gln_phase1_64pixel.yaml --phase test
# Phase 2
CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node=2 lib/main.py --cfgs ./config/gln/gln_phase2_64pixel.yaml --phase test
# # GaitGL
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --master_port 12345 --nproc_per_node=4 lib/main.py --cfgs ./config/gaitgl_64pixel.yaml --phase test
# # Baseline
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 lib/main.py --cfgs ./config/baseline_64pixel.yaml --phase test
# # SMPLGait
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 lib/main.py --cfgs ./config/smplgait_64pixel.yaml --phase test


# # ---------------------Pixel is [128, 88]--------------------- # #
# # GaitSet
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 lib/main.py --cfgs ./config/gaitset_128pixel.yaml --phase test
# # GaitPart
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 lib/main.py --cfgs ./config/gaitpart_128pixel.yaml --phase test
# # GLN
# Phase 1
#CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --master_port 12345  --nproc_per_node=2 lib/main.py --cfgs ./config/gln/gln_phase1_128pixel.yaml --phase test
# Phase 2
CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node=2 lib/main.py --cfgs ./config/gln/gln_phase2_128pixel.yaml --phase test
# # GaitGL
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --master_port 12345 --nproc_per_node=4 lib/main.py --cfgs ./config/gaitgl_128pixel.yaml --phase test
# # Baseline
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 lib/main.py --cfgs ./config/baseline_128pixel.yaml --phase test
# # SMPLGait
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 lib/main.py --cfgs ./config/smplgait_128pixel.yaml --phase test

