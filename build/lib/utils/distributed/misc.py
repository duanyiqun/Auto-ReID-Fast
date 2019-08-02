import os
import math
import torch
import multiprocessing
import torch.distributed as dist


try:
    reduce_op = dist.ReduceOp
except:
    reduce_op = dist.reduce_op


def get_world_size():
    return int(os.environ['SLURM_NTASKS'])


def get_rank():
    return int(os.environ['SLURM_PROCID'])


def get_jobid():
    return int(os.environ['SLURM_JOBID'])


def get_backend():
    return os.environ.get('DISTRIBUTED_BACKEND', None)


# work as a virtual barrier
def barrier():
    if get_world_size() > 1:
        sync_tensor = torch.ones(1).cuda()
        dist.all_reduce(sync_tensor)
        sync_tensor.item()


def all_reduce_mean(tensor_list):
    if isinstance(tensor_list, torch.Tensor):
        raise ValueError('tensor_list should be list of tensors')
    if get_world_size() == 1:
        return
    for tensor in tensor_list:
        dist.all_reduce(tensor, op=reduce_op.SUM)
        tensor.div_(get_world_size())


def all_reduce_sum(tensor_list):
    if isinstance(tensor_list, torch.Tensor):
        raise ValueError('tensor_list should be list of tensors')
    if get_world_size() == 1:
        return
    for tensor in tensor_list:
        dist.all_reduce(tensor, op=reduce_op.SUM)


def all_reduce_max(tensor_list):
    if isinstance(tensor_list, torch.Tensor):
        raise ValueError('tensor_list should be list of tensors')
    if get_world_size() == 1:
        return
    for tensor in tensor_list:
        dist.all_reduce(tensor, op=reduce_op.MAX)


def all_reduce_min(tensor_list):
    if isinstance(tensor_list, torch.Tensor):
        raise ValueError('tensor_list should be list of tensors')
    if get_world_size() == 1:
        return
    for tensor in tensor_list:
        tensor.neg_()
        dist.all_reduce(tensor, op=reduce_op.MAX)
        tensor.neg_()


def broadcast(tensor_list, src):
    if isinstance(tensor_list, torch.Tensor):
        raise ValueError('tensor_list should be list of tensors')
    if get_world_size() == 1:
        return
    for tensor in tensor_list:
        dist.broadcast(tensor, src)


def all_gather_cat(tensor_list, cat_dim=0):
    if isinstance(tensor_list, torch.Tensor):
        raise ValueError('tensor_list should be list of tensors')
    world_size = get_world_size()
    if world_size == 1:
        return tensor_list
    result_list = []
    for tensor in tensor_list:
        gather_list = [tensor.new(tensor.size()) for _ in range(world_size)]
        torch.distributed.all_gather(gather_list, tensor)
        result_list.append(torch.cat(gather_list, cat_dim))
    return result_list


def dist_segment(full_size, world_size=None, rank=None):
    if world_size is None:
        world_size = get_world_size()
    if rank is None:
        rank = get_rank()
    interval = math.ceil(full_size / world_size)
    offset = interval * rank
    part_size = min(full_size, offset + interval) - offset
    return offset, part_size


def dist_init(port, backend, mp_method='spawn'):
    os.environ['DISTRIBUTED_BACKEND'] = backend
    # start_method默认是fork，不会重新读取dataset源码，但是多机会卡死
    # 设置为spawn之后，多机不会卡死了，但是会重新读取dataset源码
    # 为了避免修改源码引发错误，将整个code目录拷贝一份副本，然后只运行副本
    if multiprocessing.get_start_method(allow_none=True) != mp_method:
        multiprocessing.set_start_method(mp_method, force=True)
    rank = get_rank()
    world_size = get_world_size()
    node_list = os.environ['SLURM_NODELIST']
    num_gpus = torch.cuda.device_count()
    gpu_id = rank % num_gpus
    torch.cuda.set_device(gpu_id)

    if world_size == 1:
        rank, world_size = 0, 1
        print('using single card, no distributed environment init', flush=True)
    else:
        if '[' in node_list:
            beg = node_list.find('[')
            pos1 = node_list.find('-', beg)
            if pos1 < 0:
                pos1 = 1000
            pos2 = node_list.find(',', beg)
            if pos2 < 0:
                pos2 = 1000
            node_list = node_list[:min(pos1, pos2)].replace('[', '')
        addr = node_list[8:].replace('-', '.')

        os.environ['MASTER_PORT'] = str(port)
        os.environ['MASTER_ADDR'] = addr
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['RANK'] = str(rank)
        dist.init_process_group(backend=backend)

        form = '%%%dd' % len(str(world_size))
        print('world_size %d, distributed init rank %s, gpu %d, at %s:%d' % (
            world_size, form % rank, gpu_id, addr, port
        ), flush=True)

    return rank, world_size
