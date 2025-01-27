import os
import glob

def count_images_in_directory(directory, extensions=None):
    if extensions is None:
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.tiff']
    
    image_count = 0
    
    for ext in extensions:
        image_count += len(glob.glob(os.path.join(directory, ext)))
    
    return image_count

if __name__ == "__main__":
    # Set the directory you want to count images in
    directory = '/home/users/pranav.rao/CT-to-Cray-Generation/bone_supress_data/im_sz_512/400/nlst'
    directory2 = '/home/users/pranav.rao/CT-to-Cray-Generation/bone_supress_data/im_sz_512/500/nlst'
    #directory3 = '/local_storage/aryan_train/sr_images'
    # directory4 = '/local_storage/aryan_training_data/original'



    # Call the function and print the result
    num_images = count_images_in_directory(directory)
    num_images2 = count_images_in_directory(directory2)
    #num_images3 = count_images_in_directory(directory3)
    #num_images4 = count_images_in_directory(directory4)
    print(f"Number of image files in '{directory}': {num_images}")
    print(f"Number of image files in '{directory2}': {num_images2}")
    #print(f"Number of image files in '{directory3}': {num_images3}")
    # print(f"Number of image files in '{directory4}': {num_images4}")



#Number of image files in '/local_storage/aryan_training_data/original': 14844
#Number of image files in '/local_storage/aryan_training_data/sr': 14844
#Number of image files in '/local_storage/aryan_training_data/1024': 14844
    
