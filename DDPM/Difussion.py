import torch
import numpy as np

# Define a linear alpha schedule for simplicity
alpha_schedule = np.linspace(0.001, 1.0, 1000)

def forward_diffusion(img, dim, alpha_schedule):
    """
    Reshape the image tensor based on the dim parameter.
    
    Parameters:
    img: torch.Tensor - The input image tensor.
    dim: tuple - The dimensions to reshape the image tensor.
    alpha_schedule: list - The list of alpha values for each timestep.
    
    Returns:
    torch.Tensor - The image tensor after applying forward diffusion and the current timestep.
    """
    # Reshape the image tensor based on the dim parameter
    img = img.reshape(-1, *dim)
    
    # Normalize the image to be in the range [-1, 1]
    img = 2 * img - 1
    
    # Get the previous time step of the image
    prev_img = img
    
    # Get the time step
    t = torch.randint(1, len(alpha_schedule), (1,)).item()
    
    # Get the noise at timestep t
    noise = torch.randn_like(img)
    
    # Get alpha_t for the current timestep
    alpha_t = alpha_schedule[t]
    
    # Add the noise to the image
    img = torch.sqrt(alpha_t) * prev_img + torch.sqrt(1 - alpha_t) * noise

    return img, t

def reverse_diffusion(img, dim, alpha_schedule, beta_schedule):

    """
    Reshape the image tensor based on the dim parameter.
    
    Parameters:
    img: torch.Tensor - The input image tensor.
    dim: tuple - The dimensions to reshape the image tensor.
    alpha_schedule: list - The list of alpha values for each timestep.
    beta_t: float - The value of beta_t for the current timestep (default is 0.0001).
    
    Returns:
    torch.Tensor - The image tensor after applying reverse diffusion.
    """

    # Reshape the image tensor based on the dim parameter
    img = img.reshape(-1, *dim)

    #get the previous time step of the image
    first_term = torch.sqrt(1 - beta_t) * img

    # Get the time step
    t = torch.randint(1, len(alpha_schedule), (1,)).item()

    # Get the noise at timestep t
    noise = torch.randn_like(img)

    # Get alpha_t for the current timestep
    alpha_t = alpha_schedule[t]

    # Get beta_t for the current timestep
    beta_t = beta_schedule[t]

    # Add the noise to the image
    img = first_term + torch.sqrt(beta_t / alpha_t) * noise

    return img