import torch
from torch import nn



def evaluate(test_data_loader, model, device, batch_size):
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
            images = images.view(-1)
            output = torch.reshape(output, (batch_size, 256, 3, 32, 32))
            output = output.permute(0,2,3,4,1).contiguous().view(-1, 256)
            loss.append(nn.functional.cross_entropy(output, (images*255).long()))

    loss=torch.Tensor(loss)   
    return torch.mean(loss)