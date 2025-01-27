import subprocess

def get_folder_size(path):
    result = subprocess.run(['du', '-sh', path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode == 0:
        return result.stdout.split()[0]
    else:
        print(f"Error: {result.stderr}")
        return None

folder_path = '/local_storage/aryan_training_data/original'
size = get_folder_size(folder_path)
if size:
    print(f"Size of folder {folder_path}: {size}")


#Size of folder /local_storage/aryan_training_data/1024: 5.9G
    
#Size of folder /local_storage/aryan_training_data/sr: 4.3G

#Size of folder /local_storage/aryan_training_data/original: 13G



