import torch
import numpy as np
import matplotlib.pyplot as plt

def sample(model, num_samples=100, device='cuda'):
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    image_size = 28*28 # MNIST dataset image size
    
    with torch.no_grad():  # Disable gradient computation
        samples = torch.zeros((num_samples, image_size), device=device)
        for i in range(image_size):
            logits = model(samples)  # Get the logits for the current samples
            probs = logits[:, i]  # Convert logits to probabilities (already done in netwoks.py as a layer)
            samples[:, i] = torch.bernoulli(probs)  # Sample from the Bernoulli distribution

    return samples.cpu().numpy().reshape(-1, 28, 28) # Reshape to (num_samples, 28, 28)

def save_samples(samples, filename='samples.png'):
    # Get the number of samples
    num_samples = samples.shape[0]

    if(num_samples<=5):
        grid_size_1 = 1
        grid_size_2 = num_samples
    else:
        # We use the ceiling of the square root of the number of samples to ensure all samples fit
        grid_size_1 = int(np.ceil(np.sqrt(num_samples)))
        grid_size_2 = grid_size_1
    
    fig, axs = plt.subplots(grid_size_1, grid_size_2, figsize=(28, 28))
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