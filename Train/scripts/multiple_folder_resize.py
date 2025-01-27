import os
from PIL import Image

def resize_image(image_path, output_path, size=(1024, 1024)):
    """Resize an image to the specified size using bicubic interpolation and save it."""
    with Image.open(image_path) as img:
        resized_img = img.resize(size, Image.BICUBIC)
        resized_img.save(output_path)

def gather_images(source_folder, output_folder, size=(1024, 1024)):
    """Gather all images from subfolders, resize them, and save to the output folder."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for root, _, files in os.walk(source_folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                source_path = os.path.join(root, file)
                output_path = os.path.join(output_folder, file)
                resize_image(source_path, output_path, size)

if __name__ == "__main__":
    source_folder = '/local_storage/pranav.rao/ct2xr/cxrs/safe_cache_body_masked/'  # Replace with the path to your source folder
    output_folder = '/local_storage/aryan_training_data/lows_res'  # Replace with the path to your output folder
    gather_images(source_folder, output_folder)
