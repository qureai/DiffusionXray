import torch
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image, to_grayscale, to_tensor
from PIL import Image
import os
import time
import torchvision.transforms as transforms

from cheff.sr.sampler import CheffSRModel

print("check : imports")

def load_and_preprocess_images(image_paths, target_size=(1024, 1024), device='cuda'):
    preprocess = transforms.Compose([
        transforms.Resize(target_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])

    batch = []
    for img_path in image_paths:
        img = Image.open(img_path)
        img_tensor = preprocess(img).unsqueeze(0)  # Add batch dimension
        batch.append(img_tensor)

    batch_tensor = torch.cat(batch, dim=0).to(device)
    return batch_tensor


device = 'cuda'
sr_path = '/raid/data_transfer/checkpoints_2/model_epoch_250_step_25000.pt'
input_folder = '/raid/data_transfer/input_test_folder/check_png'
output_folder = '/raid/data_transfer/input_test_folder/check_png_out'
batch_size = 8

print("check : loaded files")

cheff_sr = CheffSRModel(model_path=sr_path, device=device)

print("check : model loaded")

os.makedirs(output_folder, exist_ok=True)

print("check : start")

image_paths = [os.path.join(input_folder, f) for f in os.listdir(input_folder) 
               if f.endswith('.jpg') or f.endswith('.png')]

for i in range(0, len(image_paths), batch_size):
    batch_paths = image_paths[i:i+batch_size]
    filenames = [os.path.basename(path) for path in batch_paths]
    
    start_time = time.time()

    try:
        batch_tensor = load_and_preprocess_images(batch_paths, target_size=(1024, 1024), device=device)
        img_sr_batch = cheff_sr.sample(img=batch_tensor, method='ddpm')
        
        for j, img_sr in enumerate(img_sr_batch):
            grid = make_grid(img_sr.cpu())
            output_image = to_pil_image(grid)
            output_image.save(os.path.join(output_folder, f'super_resolved_{filenames[j]}'))

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Time taken to process batch {i//batch_size + 1}: {elapsed_time:.2f} seconds")

    except Exception as e:
        print(f"An error occurred while processing batch {i//batch_size + 1}: {str(e)}")

print("Processing completed.")
