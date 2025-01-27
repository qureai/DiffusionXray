import os
import shutil
import logging

def copy_png_images(src_dir, dest_dir):
    logging.info(f"Starting to copy PNG images from {src_dir} to {dest_dir}")
    
    # Walk through the source directory
    for root, dirs, files in os.walk(src_dir):
        logging.debug(f"Exploring directory: {root}")
        for file in files:
            if file.lower().endswith('.png'):
                # Construct full file path
                file_path = os.path.join(root, file)
                # Copy the file to the destination directory
                shutil.copy(file_path, dest_dir)
                logging.info(f"Copied: {file_path} to {dest_dir}")
    
    logging.info("Completed copying PNG images.")

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Define the source directory (where to look for PNG files)
    src_directory = '/raid/ct2xr/safe_cache_body_masked_1024/'
    
    # Define the destination directory (where to copy the PNG files)
    dest_directory = '/raid/data_transfer/low_res_latest'
    
    # Create the destination directory if it doesn't exist
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
        logging.info(f"Created destination directory: {dest_directory}")
    
    # Run the function
    copy_png_images(src_directory, dest_directory)
