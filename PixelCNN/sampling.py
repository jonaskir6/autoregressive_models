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
                # Get the logits for the current samples
                out = model(samples)
                # Convert logits to probabilities (already done in networks.py as a Softmax layer)
                out = F.softmax(out, dim=1)
                print(out)
                # print(probs[1][:])
                # Sample from the distribution
                probs = out[:, :, i, j]

                for k in range(3): 
                    pixel = torch.multinomial(probs[:, k], 1)
                    # print(pixel)
                    samples[:, k, i, j] = pixel[:, 0]

    return samples.permute(0,2,3,1).cpu().detach().numpy() 

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
    
    fig, axs = plt.subplots(grid_size_1, grid_size_2, figsize=(32, 32))
    axs = axs.flatten()

    for img, ax in zip(samples, axs):
        ax.imshow((img / 3 * 255).astype(int))
        ax.axis('off')

    # Turn off remaining empty subplots
    for i in range(num_samples, len(axs)):
        axs[i].axis('off')

    plt.tight_layout()
    plt.savefig(filename)
    plt.show()