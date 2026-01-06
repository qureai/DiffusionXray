import numpy as np
import os
import pandas as pd
import cv2
from matplotlib import pyplot as plt
import pickle
from PIL import Image, ImageOps
from torchvision.transforms import Resize, InterpolationMode

# Load the CSV file
df_lms = pd.read_csv("/local_storage/cxr_data/training/csvs/training_v4_lms_16-4-22.csv", index_col=0)

# Load the list of training images
train_set = pickle.load(open("/local_storage/cxr_data/file_pickles/train_images.pkl", "rb"))
train_set = [x.replace(".png", "") for x in train_set]

# Filter the DataFrame to only include rows that are in the training set
df_lms_filt = df_lms.loc[list(set(train_set).intersection(set(df_lms.index)))]

# Further filter to only include rows where "solitary" is 1
solitary_df = df_lms_filt[df_lms_filt["solitary"] == 1]

# Get the indices of the filtered DataFrame
ls = solitary_df.index

# Define the base paths for images and annotations
base_path_images = "/local_storage/cxr_data/training/images" 
base_path_annotations = "/local_storage/cxr_data/training/annotations/lms_annotations/solitary"

# Define the output path for masks
output_path = "/local_storage/aryan_training_data/mask" 
os.makedirs(output_path, exist_ok=True)

# Define the downsample transform
downsample_1024 = Resize(1024, interpolation=InterpolationMode.BICUBIC)

# Iterate over the indices and save the masks
for idx in range(len(ls)):
    img_id = ls[idx]
    
    # Load the mask image
    mask_img = cv2.imread(f"{base_path_annotations}/{img_id}.png", 0)
    
    if mask_img is not None:
        # Convert mask to PIL Image
        mask_pil = Image.fromarray(mask_img)
        
        # Downsample the mask to 1024x1024
        mask_downsampled = downsample_1024(mask_pil)
        
        # Convert back to numpy array
        mask_downsampled_np = np.array(mask_downsampled)
        
        # Save the downsampled mask image to the output folder
        output_filepath = os.path.join(output_path, f"{img_id}.png")
        cv2.imwrite(output_filepath, mask_downsampled_np)
        print(f"Saved downsampled mask for {img_id} to {output_filepath}")
    else:
        print(f"Mask for {img_id} not found.")

print("All downsampled masks have been saved.")
