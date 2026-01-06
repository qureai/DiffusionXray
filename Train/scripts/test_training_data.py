# import os
# import random
# import shutil

# def select_and_copy_images(source_dir, dest_dir, num_images=5):
#     # Create destination directory if it doesn't exist
#     if not os.path.exists(dest_dir):
#         os.makedirs(dest_dir)
    
#     # Get a list of all files in the source directory
#     files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]
    
#     # Filter the list to include only image files (you can add more extensions if needed)
#     image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']
#     images = [f for f in files if any(f.lower().endswith(ext) for ext in image_extensions)]
    
#     # Check if there are enough images in the source directory
#     if len(images) < num_images:
#         raise ValueError("Not enough images in the source directory to select from.")
    
#     # Randomly select the specified number of images
#     selected_images = random.sample(images, num_images)
    
#     # Copy the selected images to the destination directory
#     for image in selected_images:
#         src_path = os.path.join(source_dir, image)
#         dest_path = os.path.join(dest_dir, image)
#         shutil.copy(src_path, dest_path)
    
#     print(f"Copied {num_images} images to {dest_dir}")



# # Example usage
# source_directory1 = '/local_storage/aryan_training_data/training1/1024'
# source_directory2 = '/local_storage/aryan_training_data/training1/sr'
# destination_directory1 = '/local_storage/aryan_training_data/training_test/1024'
# destination_directory2 = '/local_storage/aryan_training_data/training_test/sr'
# select_and_copy_images(source_directory1, destination_directory1, num_images=5)
# select_and_copy_images(source_directory2, destination_directory2, num_images=5)

import os

def rename_images(directory):
    # Get a list of all files in the directory
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    
    # Filter the list to include only .png files
    png_files = [f for f in files if f.lower().endswith('.png')]
    
    # Check if there are exactly 5 .png images
    if len(png_files) != 5:
        raise ValueError("The directory must contain exactly 5 .png images.")
    
    # Sort the files to ensure consistent order
    png_files.sort()
    
    # Rename the images to 1.png, 2.png, 3.png, 4.png, 5.png
    for index, filename in enumerate(png_files, start=1):
        new_name = f"{index}.png"
        src_path = os.path.join(directory, filename)
        dest_path = os.path.join(directory, new_name)
        os.rename(src_path, dest_path)
    
    print("Renamed images to 1.png, 2.png, 3.png, 4.png, 5.png")

# Example usage
directory1 = '/local_storage/aryan_training_data/training_test/sr'
directory2 = '/local_storage/aryan_training_data/training_test/1024'

rename_images(directory1)
rename_images(directory2)
