import torch

def check_checkpoint(file_path):
    try:
        # Load the checkpoint
        checkpoint = torch.load(file_path)
        
        # Initialize a dictionary to hold the presence of keys
        checkpoint_info = {
            'model_state_dict': False,
            'optimizer_state_dict': False,
            'epoch': False,
            'loss': False
        }
        
        # Check for each key in the checkpoint
        if 'model_state_dict' in checkpoint:
            checkpoint_info['model_state_dict'] = True
        
        if 'optimizer_state_dict' in checkpoint:
            checkpoint_info['optimizer_state_dict'] = True
        
        if 'epoch' in checkpoint:
            checkpoint_info['epoch'] = True
        
        if 'loss' in checkpoint:
            checkpoint_info['loss'] = True
        
        # Print the results
        for key, exists in checkpoint_info.items():
            print(f"{key}: {'Present' if exists else 'Not Present'}")
    
    except Exception as e:
        print(f"An error occurred while loading the checkpoint: {e}")

# Example usage
file_path = '/home/users/aryan.goyal/chexray-diffusion/cheff_sr_fine.pt'  # Replace with your .pt file path
check_checkpoint(file_path)
