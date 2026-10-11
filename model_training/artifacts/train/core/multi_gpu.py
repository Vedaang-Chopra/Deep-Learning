from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import os
import torch


# -----------------------------------------------------------------------------
# simple launch:
# python train_gpt2.py
# DDP launch for e.g. 8 GPUs:
# torchrun --standalone --nproc_per_node=8 train_gpt2.py


## The goal is simple we just need to setup the model properly so that it runs perfectly via DDP
def setup_ddp(rank=None, world_size=None):
    """
    Sets up DDP. Can be called directly by torch.multiprocessing.spawn (Notebook)
    or implicitly via torchrun (Terminal CLI).
    """
    # set up DDP (distributed data parallel).
    # torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
    ddp = (rank is not None) or (int(os.environ.get('RANK', -1)) != -1)
    ## Not a good way to detect

    if ddp:
        # use of DDP atm demands CUDA, we set the device appropriately according to rank
        assert torch.cuda.is_available(), "CUDA is required for DDP in this configuration."
        
        
        # 2. Extract or set environment variables for manual spawning
        if rank is not None:
            # Running inside a Jupyter Notebook via mp.spawn
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '12355' # Any free port
            
            ddp_rank = rank
            ddp_local_rank = rank
            ddp_world_size = world_size
            
            # Populate environment variables so inner library calls don't break
            os.environ['RANK'] = str(rank)
            os.environ['LOCAL_RANK'] = str(rank)
            os.environ['WORLD_SIZE'] = str(world_size)
        else:
            # Multiple GPU's have different RANK
            # GPU-0-> Rank-0, GPU-1-> Rank-1, .... 
            ddp_rank = int(os.environ['RANK'])
            
            # Rank of GPU on a single node (for multi node is more important)
            ddp_local_rank = int(os.environ['LOCAL_RANK'])
            
            # World Size -> Depending on GPU (? )
            ddp_world_size = int(os.environ['WORLD_SIZE'])
            
            
        
        init_process_group(backend='nccl')
        
        
        ## Make sure to set according to cuda:<gpu_number> to make sure no collision happens 
        device = f'cuda:{ddp_local_rank}'
        
        torch.cuda.set_device(device)
        
        # Main Process, Orchestrator (Like logging, Prints etc.)
        master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    else:
        # vanilla, non-DDP run
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        # attempt to autodetect device
        device = "cpu"
        # if torch.cuda.is_available():
        #     device = "cuda"
        # elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        #     device = "mps"
        # print(f"using device: {device}")
        ## Accelerator for Pytorch
        device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    
    return {
        'ddp':ddp, 
        'device':device, 
        'ddp_rank':ddp_rank, 
        'ddp_local_rank':ddp_local_rank, 
        'ddp_world_size':ddp_world_size, 
        'master_process':master_process
    }