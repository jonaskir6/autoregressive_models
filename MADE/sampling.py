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
            probs = torch.sigmoid(logits[:, i])  # Convert logits to probabilities
            samples[:, i] = torch.bernoulli(probs)  # Sample from the Bernoulli distribution

    return samples.cpu().numpy().reshape(-1, 28, 28) # Reshape to (num_samples, 28, 28)

def save_samples(samples, filename='samples.png'):
    fig, axs = plt.subplots(10, 10, figsize=(28, 28))
    axs = axs.flatten()

    for img, ax in zip(samples, axs):
        ax.imshow(img, cmap='gray', interpolation='none')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(filename)
    plt.show()