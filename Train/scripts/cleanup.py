import os
import logging

# Configure logging
logging.basicConfig(
    filename='image_cleanup_st.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Define the folder path
folder_path = '/raid/data_transfer/low_res_latest'

# Initialize a counter for the number of images removed
removed_count = 0

# Log the start of the operation
logging.info('Started removing images with "_st_" in the filename.')

# Iterate over the files in the folder
for filename in os.listdir(folder_path):
    if '_st_' in filename:
        # Construct the full file path
        file_path = os.path.join(folder_path, filename)
        
        # Remove the file
        os.remove(file_path)
        
        # Increment the counter
        removed_count += 1
        
        # Log the removal of the file
        logging.info(f'Removed file: {filename}')

# Log the total number of images removed
logging.info(f'Number of images removed: {removed_count}')

# Print the number of images removed
print(f'Number of images removed: {removed_count}')
