import os
import random
import shutil

def select_random_images(input_dir, output_dir, num_images=100):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get list of all files in the input directory
    all_files = os.listdir(input_dir)
    
    # Filter out only image files (you can add more extensions if needed)
    image_files = [file for file in all_files if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'))]
    
    # Randomly select the specified number of images
    selected_images = random.sample(image_files, min(num_images, len(image_files)))

    # Copy selected images to the output directory
    for image in selected_images:
        src_path = os.path.join(input_dir, image)
        dst_path = os.path.join(output_dir, image)
        shutil.copy(src_path, dst_path)

    print(f'Successfully copied {len(selected_images)} images to {output_dir}')

# Set the input and output directories
input_directory = '/raid/data_transfer/UNIT/100k/100k_hr'
output_directory = '/raid/data_transfer/chexray-diffusion/Train/300k_hr_small'

# Run the function
select_random_images(input_directory, output_directory)
