import os
from typing import Dict, Final, List
import torch
import gc
import torch.distributed as dist
import torch.multiprocessing as mp
import wandb
from torch import Tensor
from torchmetrics import MeanMetric
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam, Optimizer
from torchvision.transforms import Compose, Grayscale, Lambda, ToTensor
from tqdm import tqdm
from PIL import Image

from diffusion.controller import DiffusionController
from diffusion.data import MaCheXDataset
from diffusion.diffusor import SR3DDIMDiffusor
from diffusion.utils import plot_image


class CustomSRDataset(Dataset):
    def __init__(self, root: str, transforms: Compose) -> None:
        self.root = root
        self.transforms = transforms

        
        self.hr_dir = os.path.join(root, 'hr_data')
        
        
        # Get image names from all directories
        sr_image_names = set(os.listdir(self.sr_dir))
        hr_image_names = set(os.listdir(self.hr_dir))
        mask_image_names = set(os.listdir(self.mask_dir))

        # Find common image names
        common_image_names = sr_image_names & hr_image_names & mask_image_names
        self.image_names = sorted(common_image_names)
        
        # Print the number of common images
        print(f"Number of common images across all three directories: {len(self.image_names)}")

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx: int) -> Dict[str, Image.Image]:
        img_name = self.image_names[idx]

        sr_img_path = os.path.join(self.sr_dir, img_name)
        hr_img_path = os.path.join(self.hr_dir, img_name)
        

        

        sr_img = Image.open(sr_img_path).convert('L')  # Assuming grayscale images
        hr_img = Image.open(hr_img_path).convert('L')  # Assuming grayscale images


        sr_img = self.transforms(sr_img)
        hr_img = self.transforms(hr_img)
       

        return {'LR': sr_img, 'HR': hr_img, 'SR': sr_img}

BASE_CONFIG: Final = {
    'epochs': 1000,
    'batch_size': 4,
    'num_workers': 16,
    'learning_rate': 5e-5,
    'print_freq': 50,
    'save_freq': 1000,
    'sample_freq': 5000,
    'sample_size': 4,
    'resume_checkpoint': '/raid/data_transfer/chexray-diffusion/cheff_sr_fine.pt', #path to non fine tuned 
    'data_root': '/raid/data_transfer',
    'log_dir': '/raid/data_transfer/chexray-diffusion/Train/checkpoints_mask',
    'checkpoint_dir': 'checkpoints',
    'model_params': {
        'dim': 16,
        'channels': 2,
        'out_dim': 1,
        'dim_mults': (1, 2, 4, 8, 16, 32, 32, 32),
    },
    'schedule_params': {
        'name': 'cosine',
        'timesteps': 2000,
    },
    'diffusor_params': {},
    'loss_func': 'l1',
}

def run(rank: int, world_size: int, cfg: Dict) -> None:
    """Kick off training."""
    setup(rank, world_size)
    device = torch.device(rank)

    if rank == 0:
        wandb.init(project='pre-train_ddpm_mask-9aug', config=cfg)

    # ----------------------------------------------------------------------------------
    # CREATE DIFFUSION MODEL
    # ----------------------------------------------------------------------------------
    diff = DiffusionController(
        model_params=cfg['model_params'],
        schedule_params=cfg['schedule_params'],
        diffusor_params=cfg['diffusor_params'],
        device=device,
        loss_func=cfg['loss_func'],
        ddp=True,
    )

    diff.model = DDP(diff.model, device_ids=[rank])

    optimizer = Adam(diff.get_model_params(), lr=cfg['learning_rate'])

    # Skip checkpoint loading to initialize from scratch
    print("Initializing model and optimizer from scratch.")
    cfg['resume_ep'] = 0
    cfg['resume_steps'] = 0

    # ----------------------------------------------------------------------------------
    # DATASETS & DATALOADERS
    # ----------------------------------------------------------------------------------
    transforms = Compose([Grayscale(), ToTensor(), Lambda(rescale)])

    train_dataset = CustomSRDataset(
        root=cfg['data_root'],
        transforms=transforms,
    )

    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg['batch_size'],
        sampler=train_sampler,
        persistent_workers=True,
        num_workers=cfg['num_workers'],
        pin_memory=True,
    )

    # ----------------------------------------------------------------------------------
    # CONDUCT TRAINING
    # ----------------------------------------------------------------------------------
    train(
        diff=diff,
        optimizer=optimizer,
        train_loader=train_loader,
        epochs=cfg['epochs'],
        rank=rank,
        world_size=world_size,
        cfg=cfg,
    )

def rescale(x: Tensor) -> Tensor:
    """Rescale image tensor from [0, 1] to [-1, 1]."""
    return (x * 2) - 1

def train(
    diff: DiffusionController,
    optimizer: Optimizer,
    train_loader: DataLoader,
    epochs: int,
    rank: int,
    world_size: int,
    cfg: Dict,
) -> None:
    """Execute training procedure."""
    device = diff.device
    scaler = GradScaler()

    metric_train_loss = MeanMetric().to(device)
    metrics: Dict[str, List] = {
        'train/loss': [],
    }

    log_dir = cfg['log_dir']
    checkpoint_dir = cfg['checkpoint_dir']
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    steps = 0
    ep_start = 1
    for ep in range(ep_start, epochs + 1):
        print("Epoch number :" , ep)

        for batch in tqdm(train_loader, leave=False):
            x_hr = batch['HR'].to(device)
            
            

            t = torch.randint(
                0, diff.schedule.timesteps, (x_hr.shape[0],), device=device
            ).long()

            noise = torch.randn_like(x_hr)
            x_hr_noisy = diff.diffusor.q_sample(x_start=x_hr, t=t, noise=noise)

            
            with autocast():
                predicted_noise = diff.model(x_hr_noisy, t)
                batch_loss = diff.loss_func(noise, predicted_noise)

            optimizer.zero_grad()
            scaler.scale(batch_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            metric_train_loss(batch_loss)
            steps += 1

            if steps % cfg['print_freq'] == 0 and rank == 0:
                tqdm.write(
                    'STEP {:7} | BATCH LOSS: {:.3f}'.format(steps, float(batch_loss))
                )
                wandb.log(
                    {'batch_loss': float(batch_loss), 'epoch': ep, 'step': steps},
                    step=steps,
                )

            if steps % cfg['save_freq'] == 0 and rank == 0:
                torch.save(
                    {
                        'model': diff.model.module.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': ep,
                        'steps': steps,
                    },
                    os.path.join(log_dir, f'model_epoch_{ep+1}_step_{steps}.pt'),
                )
                tqdm.write('---> CHECKPOINT SAVED <--- ')
                print("checkpoint saved")

        ep_train_loss = float(metric_train_loss.compute())
        metric_train_loss.reset()

        metrics['train/loss'].append(ep_train_loss)

        if rank == 0:
            print('EP: {:3} | LOSS: T {:.3f} '.format(ep, ep_train_loss))

        cleanup()

def save_sampling_grid(imgs: Tensor, save_path: str) -> None:
    """Save a grid of samples to a file."""
    imgs = imgs.detach().cpu()
    n_cols = imgs.shape[0]
    imgs = imgs.permute(1, 0, 2, 3, 4).reshape(-1, 1, 1024, 1024)
    plot_image(imgs, fig_size=(25, 25), ncols=n_cols, show=False, save_path=save_path)

def setup(rank: int, world_size: int) -> None:
    """Initialize environment for DDP training."""
    os.environ['MASTER_ADDR'] = 'localhost'
    try:
        default_port = os.environ['SLURM_JOB_ID']
        default_port = default_port[-4:]
        default_port = int(default_port) + 15000  # type: ignore
    except Exception:
        default_port = 12910  # type: ignore

    os.environ['MASTER_PORT'] = str(default_port)
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    print('DISTRIBUTED WORKER {} of {} INITIALIZED.'.format(rank + 1, world_size))

def is_dist_initialized():
    return dist.is_initialized()

def cleanup():
    if is_dist_initialized():
        dist.destroy_process_group()

def main() -> None:
    """Execute main func."""
    world_size = torch.cuda.device_count()
    mp.spawn(run, args=(world_size, BASE_CONFIG), nprocs=world_size)

if __name__ == '__main__':
    main()
