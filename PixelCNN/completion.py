import torch
import numpy as np
import matplotlib.pyplot as plt

def complete(model, masked_image, mask, num_samples=49, device='cuda'):
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    
    with torch.no_grad():  # Disable gradient computation
        masked_image = torch.tensor(masked_image, device=device).flatten()
        mask = torch.tensor(mask, device=device).flatten()
        
        for i in range(masked_image.size(0)):
            if mask[i] == 0:
                logits = model(masked_image)
                probs = logits[:, i]  # Convert logits to probabilities (already done in netwoks.py as a layer)
                masked_image[:, i] = torch.bernoulli(probs)  # Sample from the Bernoulli distribution

    return masked_image.cpu().numpy().reshape(-1, masked_image.shape[0], masked_image.shape[1])

def save_samples(samples, filename='samples.png', image_shape=(32, 32)):
    # Get the number of samples
    num_samples = samples.shape[0]

    if(num_samples<=5):
        grid_size_1 = 1
        grid_size_2 = num_samples
    else:
        # We use the ceiling of the square root of the number of samples to ensure all samples fit
        grid_size_1 = int(np.ceil(np.sqrt(num_samples)))
        grid_size_2 = grid_size_1
    
    fig, axs = plt.subplots(grid_size_1, grid_size_2, figsize=(image_shape[0], image_shape[1]))
    axs = axs.flatten()

    for img, ax in zip(samples, axs):
        ax.imshow(img, cmap='gray', interpolation='none')
        ax.axis('off')

    # Turn off remaining empty subplots
    for i in range(num_samples, len(axs)):
        axs[i].axis('off')

    plt.tight_layout()
    plt.savefig(filename)
    plt.show()