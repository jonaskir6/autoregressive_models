import torch
import torchvision
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import datasets, networks, sampling, evaluation
from torchvision import transforms

def get_random_image(dataset_instance):
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
   
    return image

def mask_image(image, mask_fraction=0.5):
    image_size = len(image)
    # Calculate the height of the masked area
    mask_height = int(image_size * mask_fraction)
    
    # Create a mask with ones in the top half and zeros in the bottom half
    mask = torch.ones(image_size)
    mask[-mask_height:] = 0
    
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
            completed_image = masked_image.clone()
            image_size = len(masked_image)

            for i in range(image_size):
                if mask[i] == 0:
                    logits = model(completed_image)  # Get the logits for the current samples
                    probs = logits[i]  # Convert logits to probabilities (already done in netwoks.py as a layer)
                    completed_image[i] = torch.bernoulli(probs)
            completed_images.append(completed_image.cpu().numpy().reshape(28,28))
    
    return np.array(completed_images)

def plot_completed_images(original_image, masked_image, completed_images):
    num_images = completed_images.shape[0]
    

    original_image = original_image.reshape(28,28)
    masked_image = masked_image.reshape(28,28)
   

    fig, axes = plt.subplots(1, num_images + 2, figsize=(15, 5))
    
    # Plot original image
    axes[0].imshow(original_image.cpu().numpy(),cmap='gray', interpolation='none')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Plot masked image
    axes[1].imshow(masked_image.cpu().numpy(),cmap='gray', interpolation='none')
    axes[1].set_title('Masked Image')
    axes[1].axis('off')
    
    # Plot completed images
    for i in range(num_images):
        axes[i + 2].imshow(completed_images[i],cmap='gray', interpolation='none')
        axes[i + 2].set_title(f'Completed Image {i+1}')
        axes[i + 2].axis('off')
    
    plt.tight_layout()
    plt.show()