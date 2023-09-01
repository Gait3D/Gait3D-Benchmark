
import os
import argparse
import torch
import torch.nn as nn
from modeling import models
from utils import config_loader, get_ddp_module, init_seeds, params_count, get_msg_mgr

parser = argparse.ArgumentParser(description='Main program for opengait.')
parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')                        
parser.add_argument("--local_rank", type=int,
                    help='local rank for DistributedDataParallel')
# parser.add_argument('--local_rank', type=int, default=0,
#                     help="passed by torch.distributed.launch module")
parser.add_argument('--cfgs', type=str,
                    default='config/default.yaml', help="path of config file")
parser.add_argument('--phase', default='train',
                    choices=['train', 'test'], help="choose train or test phase")
parser.add_argument('--log_to_file', action='store_true',
                    help="log to file, default path is: output/<dataset>/<model>/<save_name>/<logs>/<Datetime>.txt")
parser.add_argument('--iter', default=0, help="iter to restore")
opt = parser.parse_args()


def initialization(cfgs, training):
    msg_mgr = get_msg_mgr()
    engine_cfg = cfgs['trainer_cfg'] if training else cfgs['evaluator_cfg']
    output_path = os.path.join('output/', cfgs['data_cfg']['dataset_name'],
                               cfgs['model_cfg']['model'], engine_cfg['save_name'])
    if training:
        msg_mgr.init_manager(output_path, opt.log_to_file, engine_cfg['log_iter'],
                             engine_cfg['restore_hint'] if isinstance(engine_cfg['restore_hint'], (int)) else 0)
    else:
        msg_mgr.init_logger(output_path, opt.log_to_file)

    msg_mgr.log_info(engine_cfg)

    seed = torch.distributed.get_rank()
    init_seeds(seed)


def run_model(cfgs, training):
    msg_mgr = get_msg_mgr()
    model_cfg = cfgs['model_cfg']
    msg_mgr.log_info(model_cfg)
    Model = getattr(models, model_cfg['model'])
    model = Model(cfgs, training)
    if training and cfgs['trainer_cfg']['sync_BN']:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if cfgs['trainer_cfg']['fix_BN']:
        model.fix_BN()
    model = get_ddp_module(model)
    msg_mgr.log_info(params_count(model))
    msg_mgr.log_info("Model Initialization Finished!")

    if training:
        Model.run_train(model)
    else:
        Model.run_test(model)


def init_distributed_mode(args):
    """ init for distribute mode """
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        print("#"*20)
        print("if 1")
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        print("#"*20)
        print("if 2")
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    args.dist_url = 'env://'

    print('#'*20)
    print(f"args.dist_backend: {args.dist_backend}")
    print(f"args.rank: {args.rank}")
    print(f"args.world_size: {args.world_size}")
    print(f"args.gpu: {args.gpu}")
    print(f"args.distributed: {args.distributed}")
    print(f"args.dist_url: {args.dist_url}")
    '''
    This is commented due to the stupid icoding pylint checking.
    print('distributed init rank {}: {}'.format(args.rank, args.dist_url), flush=True)
    '''
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()


if __name__ == '__main__':
    # init_distributed_mode(opt)

    torch.distributed.init_process_group('nccl', init_method='env://')
    if torch.distributed.get_world_size() != torch.cuda.device_count():
        raise ValueError("Expect number of availuable GPUs({}) equals to the world size({}).".format(
            torch.cuda.device_count(), torch.distributed.get_world_size()))
    
    # torch.distributed.init_process_group('nccl', init_method='env://', 
    #                                      world_size=opt.world_size, rank=opt.rank)
    # # 多机多卡
    # if torch.distributed.get_world_size() > 1:
    #     tensor_list = []
    #     for dev_idx in range(torch.cuda.device_count()):
    #         tensor_list.append(torch.FloatTensor([1]).cuda(dev_idx))
    #     torch.distributed.all_reduce_multigpu(tensor_list)
    # # 单机多卡
    # else:
    #     if torch.distributed.get_world_size() != torch.cuda.device_count():
    #         raise ValueError("Expect number of availuable GPUs({}) equals to the world size({}).".format(
    #             torch.cuda.device_count(), torch.distributed.get_world_size()))
    
    cfgs = config_loader(opt.cfgs)
    if opt.iter != 0:
        cfgs['evaluator_cfg']['restore_hint'] = int(opt.iter)
        cfgs['trainer_cfg']['restore_hint'] = int(opt.iter)

    training = (opt.phase == 'train')
    initialization(cfgs, training)
    run_model(cfgs, training)
