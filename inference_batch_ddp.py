import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image, to_grayscale, to_tensor
from PIL import Image
import os
import time

from cheff.sr.sampler import CheffSRModel

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def process_batch(rank, world_size, sr_path, input_folder, output_folder, batch_size):
    setup(rank, world_size)
    
    device = f'cuda:{rank}'
    cheff_sr = CheffSRModel(model_path=sr_path, device=device)
    cheff_sr = DDP(cheff_sr, device_ids=[rank])

    os.makedirs(output_folder, exist_ok=True)

    all_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.png'))]
    files_per_gpu = len(all_files) // world_size
    start_idx = rank * files_per_gpu
    end_idx = start_idx + files_per_gpu if rank != world_size - 1 else len(all_files)
    
    for i in range(start_idx, end_idx, batch_size):
        batch = []
        filenames = []
        
        for j in range(i, min(i + batch_size, end_idx)):
            filename = all_files[j]
            img_path = os.path.join(input_folder, filename)
            img = Image.open(img_path)
            img = to_grayscale(img)
            img = to_tensor(img)
            batch.append(img)
            filenames.append(filename)

        if batch:
            start_time = time.time()

            batch_tensor = torch.stack(batch).to(device)
            img_sr_batch = cheff_sr.module.sample(img=batch_tensor, method='DDPM')
            
            for k, img_sr in enumerate(img_sr_batch):
                grid = make_grid(img_sr.cpu())
                output_image = to_pil_image(grid)
                output_image.save(os.path.join(output_folder, f'super_resolved_{filenames[k]}'))

            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"GPU {rank}: Time taken to process batch: {elapsed_time:.2f} seconds")

    cleanup()

def run_parallel_inference(sr_path, input_folder, output_folder, batch_size):
    world_size = torch.cuda.device_count()
    print(f"Using {world_size} GPUs")
    mp.spawn(process_batch,
             args=(world_size, sr_path, input_folder, output_folder, batch_size),
             nprocs=world_size,
             join=True)

if __name__ == "__main__":
    sr_path = '/raid/data_transfer/chexray-diffusion/Train/checkpoints_ddpm_lowres_t2/model_epoch_699_step_74000.pt'
    input_folder = '/raid/data_transfer/hr_data'
    output_folder = '/raid/MUNIT_data/300k_LR_DDPM_ft'
    batch_size = 1

    run_parallel_inference(sr_path, input_folder, output_folder, batch_size)