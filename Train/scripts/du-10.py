import os
import random
import subprocess

def get_file_size(file_path):
    try:
        result = subprocess.run(['du', '-sh', file_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            return result.stdout.split()[0]
        else:
            print(f"Error getting size for {file_path}: {result.stderr}")
            return None
    except Exception as e:
        print(f"Exception getting size for {file_path}: {str(e)}")
        return None

def main():
    folder_path = '/home/users/aryan.goyal/chexray-diffusion/data_new/new_100_sr'
    
    # List all files in the directory
    all_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    
    # Ensure we have at least 10 files
    if len(all_files) < 10:
        print(f"Not enough files in {folder_path} to select 10 random files.")
        return
    
    # Select 10 random files
    random_files = random.sample(all_files, 10)
    
    # Get and print the size of each random file
    for file_path in random_files:
        size = get_file_size(file_path)
        if size:
            print(f"Size of {file_path}: {size}")

if __name__ == '__main__':
    main()
