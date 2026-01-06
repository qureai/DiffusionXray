"""Train a diffusion model for super resolution."""
import os
from typing import (
    Dict,
    Final,
    List,
)

import torch
import gc 
import torch.distributed as dist
import torch.multiprocessing as mp
import wandb
from torch import Tensor
from torchmetrics import MeanMetric
from torch.utils.data import ConcatDataset, Dataset
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader, DistributedSampler
from torchmetrics import MeanMetric
from torchvision.transforms import (
    Compose,
    Grayscale,
    Lambda,
    ToTensor,
)
from tqdm import tqdm
from PIL import Image

from diffusion.controller import DiffusionController
from diffusion.data import MaCheXDataset
from diffusion.diffusor import SR3DDIMDiffusor
from diffusion.utils import plot_image


#transfer mask images 




class CustomSRDataset(Dataset):
    def __init__(self, root: str, transforms: Compose) -> None:
        self.root = root
        self.transforms = transforms

        self.sr_dir = os.path.join(root, 'low_res_latest')
        self.hr_dir = os.path.join(root, 'low_res_latest')
        self.image_names = os.listdir(self.sr_dir)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx: int):
        img_name = self.image_names[idx]

        sr_img_path = os.path.join(self.sr_dir, img_name)
        hr_img_path = os.path.join(self.hr_dir, img_name)
        

        sr_img = Image.open(sr_img_path).convert('L')  # Assuming grayscale images
        hr_img = Image.open(hr_img_path).convert('L')  # Assuming grayscale images
       

        sr_img = self.transforms(sr_img)
        hr_img = self.transforms(hr_img)
        

        return {'LR': sr_img, 'HR': hr_img, 'SR': sr_img}


#print("Environment setup complete.")

BASE_CONFIG: Final = {
    'epochs': 700,
    'batch_size': 18,
    'num_workers': 16,
    'learning_rate': 5e-5,
    'print_freq': 50,
    'save_freq': 1000,
    'sample_freq': 5000,
    'sample_size': 4,
    'resume_checkpoint': '/raid/data_transfer/cheff_sr_base.pt', #path to non fine tuned 
    'data_root': '/raid/data_transfer/',
    'log_dir': '/raid/data_transfer/chexray-diffusion/Train/checkpoints_ddpm_new_lowres',
    'checkpoint_dir': 'checkpoints', #change this 

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
        wandb.init(project='fine-tuning-ddpm-lowres-on-new-data', config=cfg)

    # ----------------------------------------------------------------------------------
    # CREATE DIFFUSION MODEL
    # ----------------------------------------------------------------------------------
    #print("creating diffusion model ..")
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

    if cfg['resume_checkpoint'] is not None and os.path.exists(cfg['resume_checkpoint']):
        state_dict = torch.load(cfg['resume_checkpoint'], map_location=device)
    
    
        if rank == 0:

            if 'optimizer' in state_dict:
                optimizer.load_state_dict(state_dict['optimizer'])
            else:
                print("Optimizer state not found in checkpoint, initializing from scratch.")

            if 'epoch' in state_dict:
                cfg['resume_ep'] = state_dict['epoch']
            else:
                print("Epoch information not found in the checkpoint, setting epoch to 0.")
                cfg['resume_ep'] = 0

            if 'steps' in state_dict:
                cfg['resume_steps'] = state_dict['steps']
            else:
                print("Steps information not found in the checkpoint, setting steps to 0.")
                cfg['resume_steps'] = 0
    
            if rank == 0:
                print('--> RETURNING FROM CHECKPOINT: EPOCH {} | STEP {}'.format(cfg['resume_ep'], cfg['resume_steps']))
    
        del state_dict
        gc.collect()

    #print("CREATE DIFFUSION MODEL done")
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
    #print("data loaded")
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


    #print("training configs done")


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

    # ----------------------------------------------------------------------------------
    # METRICS
    # ----------------------------------------------------------------------------------
    #metric_train_loss = MeanMetric(compute_on_step=False).to(device)
    metric_train_loss = MeanMetric().to(device)
    # We also define a dictionary that keeps track over all computed metrics.
    metrics: Dict[str, List] = {
        'train/loss': [],
    }

    log_dir = cfg['log_dir']
    checkpoint_dir = cfg['checkpoint_dir']
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    #print("START TRAINING PROCEDURE")
    # ----------------------------------------------------------------------------------
    # START TRAINING PROCEDURE
    # ----------------------------------------------------------------------------------
    steps = 0 if cfg.get('resume_steps') is None else cfg.get('resume_steps')
    ep_start = 1 if cfg.get('resume_ep') is None else cfg.get('resume_ep')
    for ep in range(ep_start, epochs + 1):
        print("Epoch number :" , ep)
          # type: ignore
        #print("training loop starts")
        # ------------------------------------------------------------------------------
        # TRAINING LOOP
        # ------------------------------------------------------------------------------
        for batch in tqdm(train_loader, leave=False):
            # True high resolution ground truth
            x_hr = batch['HR'].to(device)
            # Interpolated low resolution as conditioning
            #x_sr = batch['SR'].to(device)  #low_res -- use as conditioning
            #x_mask = batch['Mask'].to(device)

            # Sample random timestep
            t = torch.randint( 
                0, diff.schedule.timesteps, (x_hr.shape[0],), device=device
            ).long()

            # Create noisy HR image
            noise = torch.randn_like(x_hr)
            noise2 = torch.randn_like(x_hr)
            x_hr_noisy = diff.diffusor.q_sample(x_start=x_hr, t=t, noise=noise)
            x_hr_noisy2 = diff.diffusor.q_sample(x_start=x_hr, t=t, noise=noise2)

            # x_sr_noisy1 = diff.diffusor.q_sample(x_start=x_sr, t=t, noise=noise1)
            # x_sr_noisy2 = diff.diffusor.q_sample(x_start=x_sr, t=t, noise=noise2)

            # Concatenate noisy HR image with SR image
            #x_in = torch.cat([x_hr_noisy, x_sr], dim=1)
            x_in = torch.cat([x_hr_noisy, x_hr_noisy2], dim=1)

            #x_in = x_sr_noisy

            # Predict noise on HR image
            with autocast():
                # Predict noise on HR image
                predicted_noise = diff.model(x_in, t)
                # Compute loss
                batch_loss = diff.loss_func(noise + noise2, predicted_noise)

            optimizer.zero_grad()
            scaler.scale(batch_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            metric_train_loss(batch_loss)
            steps += 1  # type: ignore

            #print("intermediate")

            # --------------------------------------------------------------------------
            # PRINT INTERMEDIATE PROGRESS
            # --------------------------------------------------------------------------

            if steps % cfg['print_freq'] == 0 and rank == 0:
                tqdm.write(
                    'STEP {:7} | BATCH LOSS: {:.3f}'.format(steps, float(batch_loss))
                )
                wandb.log(
                    {'batch_loss': float(batch_loss), 'epoch': ep, 'step': steps},
                    step=steps,
                )

            # --------------------------------------------------------------------------
            # SAVE MODEL
            # --------------------------------------------------------------------------
            if steps % cfg['save_freq'] == 0 and rank == 0:
                torch.save(
                    {
                        'model': diff.model.module.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': ep,
                        'steps': steps,
                    },
                    #os.path.join(log_dir, 'model_922.pt'),
                    os.path.join(log_dir, f'model_epoch_{ep+1}_step_{steps}.pt'),
                )
                tqdm.write('---> CHECKPOINT SAVED <--- ')
                print("checkpoint saved")

            # --------------------------------------------------------------------------
            # EVALUATION SAMPLES
            # --------------------------------------------------------------------------
            #print("evalauation samples")
            # if steps % cfg['sample_freq'] == 0:
            #     if rank == 0:
            #         tqdm.write('---> GENERATING SAMPLES FROM MODEL <---')
            #     with autocast():
            #         diff.model.eval()
            #         diffusor = SR3DDIMDiffusor(
            #             model=diff.model, schedule=diff.schedule, device=device
            #         )
            #         x_eval = x_sr[: min(cfg['sample_size'], len(x_sr))]
            #         imgs = diffusor.p_sample_loop_with_steps(
            #             sr=x_eval, log_every_t=diff.schedule.timesteps // 4
            #         )
            #         diff.model.train()

            #     imgs = imgs.detach().float()
            #     imgs.clamp_(-1, 1)
            #     imgs = (imgs + 1) / 2

            #     tensor_list = [
            #         torch.zeros(imgs.shape, dtype=torch.float, device=device)
            #         for _ in range(world_size)
            #     ]
            #     dist.all_gather(tensor_list=tensor_list, tensor=imgs)

            #     if rank == 0:
            #         gathered_imgs = torch.cat(tensor_list, dim=1)
            #         grid_path = os.path.join(
            #             cfg['log_dir'], 'sample_step_{}'.format(str(steps).zfill(6))
            #         )
            #         save_sampling_grid(gathered_imgs, grid_path)

            #     dist.barrier()

        # ------------------------------------------------------------------------------
        # METRIC COMPUTATION
        # ------------------------------------------------------------------------------
        #print("metric computation")
        ep_train_loss = float(metric_train_loss.compute())
        metric_train_loss.reset()

        # Add current metrics to history.
        metrics['train/loss'].append(ep_train_loss)

        if rank == 0:
            print('EP: {:3} | LOSS: T {:.3f} '.format(ep, ep_train_loss))

            # Save model and exit.
            # This is due to time constraints in a SLURM Cluster.
            # A follow-up job is triggered externally to continue training.
            # torch.save(
            #     {
            #         'model': diff.model.module.state_dict(),
            #         'optimizer': optimizer.state_dict(),
            #         'epoch': ep + 1,
            #         'steps': steps,
            #     },
            #     #os.path.join(log_dir, 'model_11june_701.pt'),
            #     os.path.join(log_dir, f'model_epoch_{ep}_step_{steps}.pt'),
            # )
        #print("cleanup start")
        cleanup()
        #print("ckleanup done ")
        #return


def save_sampling_grid(imgs: Tensor, save_path: str) -> None:
    """Save a grid of samples to a file."""
    imgs = imgs.detach().cpu()
    n_cols = imgs.shape[0]
    imgs = imgs.permute(1, 0, 2, 3, 4).reshape(-1, 1, 1024, 1024)
    plot_image(imgs, fig_size=(25, 25), ncols=n_cols, show=False, save_path=save_path)


def setup(rank: int, world_size: int) -> None:
    """Initialize environment for DDP training."""
    # Set adress and port for node communication.
    # We simply choose localhost here, which should suffice in most cases.
    os.environ['MASTER_ADDR'] = 'localhost'

    # The master port needs to be free, which could be an issue in a multi-user setting.
    # as we're using slurm to deploy our jobs, we can use the last digits of the slurm
    # job id to get our port. Otherwise, we use a default.
    try:
        default_port = os.environ['SLURM_JOB_ID']
        default_port = default_port[-4:]

        # All ports should be in the 10k+ range
        default_port = int(default_port) + 15000  # type: ignore

    except Exception:
        default_port = 12910  # type: ignore

    os.environ['MASTER_PORT'] = str(default_port)

    # Initialize distributed process group.
    # torch offers a few backends, but usually NCCL is the best choice.
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    print('DISTRIBUTED WORKER {} of {} INITIALIZED.'.format(rank + 1, world_size))


# def cleanup():
#     """Clean up process groups from DDP training."""
#     #wandb.finish()
#     dist.destroy_process_group()

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
