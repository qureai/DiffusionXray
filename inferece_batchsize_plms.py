import torch
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image, to_grayscale, to_tensor
from PIL import Image
import os
import time

from cheff import CheffSRModel

print("check : imports")

device = 'cuda'
sr_path = '/home/users/aryan.goyal/chexray-diffusion/cheff_sr_fine.pt'
input_folder = '/local_storage/aryan_train/low_res'  # Specify the path to the folder containing input images
output_folder = '/local_storage/aryan_train/inference_low_res'  # Specify the path to the folder to save the output images
batch_size = 4  # Set the desired batch size

print("check : loaded files")

cheff_sr = CheffSRModel(model_path=sr_path, device=device)

print("check : model loaded")

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Accumulate images in a batch
batch = []
filenames = []
print("check : start")
# Process each image in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith('.jpg') or filename.endswith('.png'):  # Check for image files
        img_path = os.path.join(input_folder, filename)
        img = Image.open(img_path)
        img = to_grayscale(img)  # Convert to grayscale
        img = to_tensor(img)  # Convert to tensor
        batch.append(img)
        filenames.append(filename)

        # If batch is full, process it
        if len(batch) == batch_size:
            start_time = time.time()  # Start the timer

            batch_tensor = torch.stack(batch).to(device)  # Create a batch tensor
            img_sr_batch = cheff_sr.sample(img=batch_tensor, method='ddpm')
            
            for i, img_sr in enumerate(img_sr_batch):
                grid = make_grid(img_sr.cpu())
                output_image = to_pil_image(grid)
                output_image.save(os.path.join(output_folder, f'super_resolved_{filenames[i]}'))

            end_time = time.time()  # End the timer
            elapsed_time = end_time - start_time  # Calculate elapsed time
            print(f"Time taken to process batch: {elapsed_time:.2f} seconds")

            # Clear the batch and filenames
            batch = []
            filenames = []

# Process any remaining images
if batch:
    start_time = time.time()  # Start the timer

    batch_tensor = torch.stack(batch).to(device)
    img_sr_batch = cheff_sr.sample(img=batch_tensor, method='ddpm')

    for i, img_sr in enumerate(img_sr_batch):
        grid = make_grid(img_sr.cpu())
        output_image = to_pil_image(grid)
        output_image.save(os.path.join(output_folder, f'super_resolved_{filenames[i]}'))

    end_time = time.time()  # End the timer
    elapsed_time = end_time - start_time  # Calculate elapsed time
    print(f"Time taken to process remaining batch: {elapsed_time:.2f} seconds")
