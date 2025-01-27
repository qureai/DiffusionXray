import os
import random
import cv2
import numpy as np

def add_random_gaussian_noise(image, max_sigma=24):
    """Add Gaussian noise to an image with a random standard deviation."""
    #sigma = random.uniform(0, max_sigma)
    sigma = 25
    noise = np.random.normal(0, sigma, image.shape)
    noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
    return noisy_image

def process_images(input_dir, output_dir, max_sigma=24):
    """Process each image in the input directory, apply random Gaussian noise, and save to the output directory."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of all files in the input directory
    all_files = os.listdir(input_dir)
    
    # Filter out only image files (you can add more extensions if needed)
    image_files = [file for file in all_files if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    
    for image_file in image_files:
        image_path = os.path.join(input_dir, image_file)
        
        # Read the image
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        
        # Apply random Gaussian noise
        noisy_image = add_random_gaussian_noise(image, max_sigma)
        
        # Save the noisy image to the output directory
        output_path = os.path.join(output_dir, image_file)
        cv2.imwrite(output_path, noisy_image)
        
    print(f'Successfully processed {len(image_files)} images.')

# Set the input and output directories
input_directory = '/raid/data_transfer/chexray-diffusion/Train/300k_lr_small'
output_directory = '/raid/data_transfer/chexray-diffusion/Train/300k_lr_small_g3'

# Run the function
process_images(input_directory, output_directory)
