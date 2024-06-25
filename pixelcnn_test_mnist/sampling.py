import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision 

def sample(model, num_samples=100, device='cuda'):
    model.to(device)
    # Set the model to evaluation mode
    model.eval()
    
    # Disable gradient computation
    with torch.no_grad():  
        samples = torch.Tensor(num_samples, 1, 28, 28).to(device)
        samples.fill_(0)
        for i in range(28):
            for j in range(28):
                # Get the logits for the current samples
                out = model(samples)
                # Convert logits to probabilities (already done in networks.py as a Softmax layer)
                probs = F.softmax(out[:, :, i, j], dim=-1).data
                # print(probs[1][:])
                # Sample from the distribution
                samples[:, :, i, j] = torch.multinomial(probs, 1).float() / 255.0
                # print(pixel)
    torchvision.utils.save_image(samples, 'sample.png', nrow=10, padding=0)
    