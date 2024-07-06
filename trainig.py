# loss function
from Difussion import forward_diffusion, linear_beta_schedule, get_index_from_list
import torch
import matplotlib.pyplot as plt
from model import model
device = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 32
T = 200

from torchvision import transforms 
import numpy as np

def show_tensor_image(image):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])

    # Take first image of batch
    if len(image.shape) == 4:
        image = image[0, :, :, :] 
    plt.imshow(reverse_transforms(image))

def loss(model, x_0, t, device="cpu"):
    x_noisy, noise = forward_diffusion(x_0, t, device)
    predicted_noise = model(x_noisy, t)
    loss = ((noise - predicted_noise) ** 2).mean()
    return loss

@torch.no_grad()
def sample(x, t):

    betas_t = get_index_from_list(linear_beta_schedule(), t, x.shape)

    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod_t, t, x.shape
    )

    sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas_t, t, x.shape)
    # Call model (current image - noise prediction)

    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )

    posterior_variance_t = get_index_from_list(
        posterior_variance_t, t, x.shape
    )

    if t == 0:
        return model_mean
    else:
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise
    
@torch.no_grad()
def sample_plot_image():
    # Sample noise
    img_size = IMG_SIZE
    img = torch.randn((1, 3, img_size, img_size), device=device)
    plt.figure(figsize=(15,15))
    plt.axis('off')
    num_images = 10
    stepsize = int(T/num_images)

    for i in range(0,T)[::-1]:
        t = torch.full((1,), i, device=device, dtype=torch.long)
        img = sample(img, t)
        # Edit: This is to maintain the natural range of the distribution
        img = torch.clamp(img, -1.0, 1.0)
        if i % stepsize == 0:
            plt.subplot(1, num_images, int(i/stepsize)+1)
            show_tensor_image(img.detach().cpu())
    plt.show()            