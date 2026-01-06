import os

def count_common_images(sr_dir: str, hr_dir: str) -> None:
    """Count and print the number of common images across the three directories."""
    # Get image names from all directories
    sr_image_names = set(os.listdir(sr_dir))
    hr_image_names = set(os.listdir(hr_dir))
    

    # Find common image names
    common_image_names = sr_image_names & hr_image_names 

    # Print the number of common images
    print(f"Number of common images across all three directories: {len(common_image_names)}")

if __name__ == '__main__':
    # Define paths to the directories
    sr_dir = '/raid/data_transfer/munit_inference_13k_nodules_01_gb'
    hr_dir = '/raid/data_transfer/UNIT/100k/100k_hr'
    

    count_common_images(sr_dir, hr_dir)
