
import torch
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image, to_grayscale, to_tensor


from cheff import CheffSRModel

device = 'cuda'
sr_path = '/home/users/aryan.goyal/chexray-diffusion/trained_models/cheff_sr_fine.pt'

cheff_sr = CheffSRModel(model_path=sr_path, device=device)

# Convert to grayscale
imgs = torch.stack([to_tensor(to_grayscale(to_pil_image(i))) for i in imgs])

# Predict super resolution image
imgs_sr = cheff_sr.sample(
    img=imgs,
    method='ddpm'
)

grid = make_grid(imgs_sr.cpu())
to_pil_image(grid)
