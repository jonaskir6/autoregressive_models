import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

def sample(model, num_samples=100, device='cuda'):
    model.to(device)
    # Set the model to evaluation mode
    model.eval()
    
    # Disable gradient computation
    with torch.no_grad():  
        samples = torch.zeros((num_samples, 3, 32, 32), device=device)
        for i in range(32):
            for j in range(32):
                # Iterate over each channel
                for k in range(3):  
                    # Get the logits for the current samples
                    out = model(samples)
                    # Convert logits to probabilities (already done in networks.py as a Softmax layer)
                    probs = F.softmax(torch.reshape(out, (num_samples, 256, 3, 32, 32))[:, :, k, i, j], dim=1)
                    # Sample from the distribution
                    pixel = torch.multinomial(probs, 1).squeeze(1)
                    samples[:, k, i, j] = pixel / 255

    return samples.permute(0,2,3,1).cpu().detach().numpy() 

def save_samples(samples, filename='samples.png', mnist = False):
    # Get the number of samples
    num_samples = samples.shape[0]

    if(num_samples<=5):
        grid_size_1 = 1
        grid_size_2 = num_samples
    else:
        # We use the ceiling of the square root of the number of samples to ensure all samples fit
        grid_size_1 = int(np.ceil(np.sqrt(num_samples)))
        grid_size_2 = grid_size_1
    
    _, axs = plt.subplots(grid_size_1, grid_size_2, figsize=(8, 8))
    axs = axs.flatten()

    for img, ax in zip(samples, axs):
        if(mnist):
            ax.imshow(img, cmap='Greys_r')
        else:    
            ax.imshow((img * 255).astype(int))
        ax.axis('off')


    # Turn off remaining empty subplots
    for i in range(num_samples, len(axs)):
        axs[i].axis('off')

    plt.tight_layout()
    plt.savefig(filename)
    plt.show()