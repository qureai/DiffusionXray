import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch

# Define the CustomSRDataset class
class CustomSRDataset(Dataset):
    def __init__(self, root: str, transforms: transforms.Compose) -> None:
        self.root = root
        self.transforms = transforms

        self.sr_dir = os.path.join(root, 'sr_images')
        self.hr_dir = os.path.join(root, 'hr_images')
        self.mask_dir = os.path.join(root, 'masks_images')
        self.image_names = os.listdir(self.sr_dir)

    def __len__(self):
        return 1  # Load only one image

    def __getitem__(self, idx: int):
        img_name = self.image_names[idx]

        sr_img_path = os.path.join(self.sr_dir, img_name)
        hr_img_path = os.path.join(self.hr_dir, img_name)
        mask_img_path = os.path.join(self.mask_dir, img_name)

        sr_img = Image.open(sr_img_path).convert('L')  # Assuming grayscale images
        hr_img = Image.open(hr_img_path).convert('L')  # Assuming grayscale images
        mask_img = Image.open(mask_img_path).convert('L')

        sr_img = self.transforms(sr_img)
        hr_img = self.transforms(hr_img)
        mask_img = self.transforms(mask_img)

        return {'LR': sr_img, 'HR': hr_img, 'SR': sr_img, 'Mask': mask_img}

# Define the transforms
transform = transforms.Compose([
    transforms.ToTensor()
])

# Instantiate the dataset and dataloader
dataset = CustomSRDataset(root='/local_storage/aryan_train/', transforms=transform)
train_loader = DataLoader(dataset, batch_size=1, shuffle=False)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Process one batch and print shapes
batch = next(iter(train_loader))
x_hr = batch['HR'].to(device)
x_sr = batch['SR'].to(device)
mask = batch['Mask'].to(device)

# # Sample random timestep
# t = torch.randint(0, 1000, (x_hr.shape[0],), device=device).long()

# # Create noisy HR image
# noise = torch.randn_like(x_hr)
x_hr_noisy = x_hr #+ noise  # Simplified for demonstration

# Concatenate noisy HR image with SR image
x_in2 = torch.cat([x_hr_noisy, x_sr], dim=1)
x_in = torch.cat([x_hr_noisy, mask], dim=1)

print(f"x_in2 shape: {x_in2.shape}")
print(f"x_in shape: {x_in.shape}")
