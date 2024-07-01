import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np


def completemnist(model, device, images):
    img_chn = 1
    img_size = 28
    mask = torch.zeros(1, 1, img_size, img_size)
    mask[:, :, :img_size // 2, :] = 1

    samples = []

    # Convert occluded_image to tensor and reshape
    for image in images:
        sample = torch.zeros(1, img_chn, img_size, img_size).to(device)
        image = torch.tensor(image, dtype=torch.float32).view(1, img_chn, img_size, img_size).to(device)
        image = image[:, :, :img_size//2, :]
        sample[0, 0, :img_size//2, :] = image
        plt.imshow(sample[0].reshape(28, 28).to('cpu'), cmap='Greys_r')
        plt.show()

        for i in range(img_size):
            for j in range(img_size):
                if mask[0, 0, i, j] == 0:
                    out = model(sample)
                    probs = F.softmax(out[:, :, i, j], dim=-1).data
                    sample[:, :, i, j] = torch.multinomial(probs, 1).float() / 255.0
        
        samples.append(sample)

    for idx in range(len(samples)):
        plt.imshow(samples[idx][0].permute(1, 2, 0).to('cpu'), cmap='Greys_r')
        plt.show()


def completecifar(model, images, device):
    img_chn = 3
    img_size = 32
    mask = torch.zeros(1, 3, img_size, img_size)
    mask[:, :, :img_size // 2, :] = 1
    sample = torch.zeros(1, img_chn, img_size, img_size).to(device)
    
    samples = []

    # Convert occluded_image to tensor and reshape
    for image in images:
        image = torch.tensor(image, dtype=torch.float32).view(1, img_chn, img_size, img_size).to(device)
        image = image[:, :, :img_size//2, :]
        sample[0, :, :img_size//2, :] = image

        for c in range(img_chn):
            for i in range(img_size):
                for j in range(img_size):
                    if mask[0, c, i, j] == 0:
                        out = model(sample)
                        probs = F.softmax(out[:, c, i, j], dim=-1).data
                        sample[:, c, i, j] = torch.multinomial(probs, 1).float() / 255.0

        samples.append(sample)
    
    for idx in range(len(samples)):
        plt.imshow(samples[idx].permute(1, 2, 0).to('cpu'))
        plt.show()

def get_random_image(dataset_instance, count=1):
    """
    usage: 
    ds = datasets.Dataset('cifar10', batch_size=1)
    get_random_cifar10_image(ds)
    """
    # Get the test data loader from the dataset instance
    test_data_loader = dataset_instance.get_test_data_loader()
    # Convert the data loader to a list to sample a random image
    test_data_list = list(test_data_loader)

    image_list = []

    for _ in range(count):
        random_batch_index = np.random.randint(0, len(test_data_list))
        images, _ = test_data_list[random_batch_index]

        random_image_index = np.random.randint(0, images.size(0))
        image = images[random_image_index]

        image_list.append(image)

    return image_list