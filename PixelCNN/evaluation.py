import torch
from torch import nn



def evaluate(test_data_loader, model, device):
    num_correct = 0
    num_total = 0

  # Disable gradient computation.
    with torch.no_grad():
        # Switch to evaluation mode.
        model.eval()
        loss=[]
        # Iterate over the entire test dataset.
        for images, labels in test_data_loader:
       
            images = images.to(device)
            output = model(images)
            loss.append(nn.functional.cross_entropy(output,images))

    loss=torch.Tensor(loss)   
    return torch.mean(loss)