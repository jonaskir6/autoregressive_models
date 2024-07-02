import torch
import torchvision
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import datasets, networks, sampling, evaluation
from torchvision import transforms

def get_random_cifar10_image(dataset_instance):
    """
    usage: 
    ds = datasets.Dataset('cifar10', batch_size=1)
    get_random_cifar10_image(ds)
    """
    # Get the test data loader from the dataset instance
    test_data_loader = dataset_instance.get_test_data_loader()
    # Convert the data loader to a list to sample a random image
    test_data_list = list(test_data_loader)
    
    # Get a random batch of images and labels
    random_batch_index = np.random.randint(0, len(test_data_list))
    images, labels = test_data_list[random_batch_index]
    
    # Get a random image from the batch
    random_image_index = np.random.randint(0, images.size(0))
    image = images[random_image_index]
    label = labels[random_image_index]
    
    return image

def mask_image(image, mask_fraction=0.5):
    C, H, W = image.shape
    
    # Calculate the height of the masked area
    mask_height = int(H * mask_fraction)
    
    # Create a mask with ones in the top half and zeros in the bottom half
    mask = torch.ones((C, H, W))
    mask[:, -mask_height:, :] = 0
    
    # Apply the mask to the image
    masked_image = image * mask
    
    return masked_image, mask

def complete(model, masked_image, mask, num_completions=5, device='cuda'):
    model.to(device)
    masked_image = masked_image.to(device)
    mask = mask.to(device)

    model.eval()
    completed_images = []
    with torch.no_grad():
        for _ in range(num_completions):
            completed_image = masked_image.clone().unsqueeze(0)
            C, H, W = masked_image.shape

            for i in range(H):
                for j in range(W):
                    for c in range(C):
                        if mask[c, i, j] == 0:
                            out = model(completed_image)
                            # Convert logits to probabilities (already done in networks.py as a Softmax layer)
                            probs = F.softmax(torch.reshape(out, (1, 256, 3, 32, 32))[:, :, c, i, j], dim=1)
                         # Sample from the distribution
                            pixel = torch.multinomial(probs, 1).squeeze(1)
                            completed_image[:, c, i, j] = pixel / 255
            completed_images.append(completed_image.squeeze(0).cpu().numpy())
    
    return np.array(completed_images)

def plot_completed_images(original_image, masked_image, completed_images):
    num_images = completed_images.shape[0]
    
    fig, axes = plt.subplots(1, num_images + 2, figsize=(15, 5))
    
    # Plot original image
    axes[0].imshow(original_image.permute(1, 2, 0).cpu().numpy())
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Plot masked image
    axes[1].imshow(masked_image.permute(1, 2, 0).cpu().numpy())
    axes[1].set_title('Masked Image')
    axes[1].axis('off')
    
    # Plot completed images
    for i in range(num_images):
        axes[i + 2].imshow(np.transpose(completed_images[i], (1, 2, 0)))
        axes[i + 2].set_title(f'Completed Image {i+1}')
        axes[i + 2].axis('off')
    
    plt.tight_layout()
    plt.show()