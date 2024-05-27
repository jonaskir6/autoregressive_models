import torch
from torchvision import datasets, transforms    

# MNIST example from paper is in binarized mnist
class Binarize(object):
    def __call__(self, tensor):
        return (tensor > 0.5).float()

class Dataset(object):

    def __init__(self, data_dir):
        self.data_dir = data_dir
        # Binirize MNIST
        self.transform = transforms.Compose([transforms.ToTensor(),
                                              Binarize()])
        # Train and test data
        self.train_data = datasets.MNIST(self.data_dir, train=True, download=True, transform=self.transform)
        self.test_data = datasets.MNIST(self.data_dir, train=False, transform=self.transform)
        # Data loaders
        self.train_data_loader = torch.utils.data.DataLoader(self.train_data, batch_size=64, shuffle=True)
        self.test_data_loader = torch.utils.data.DataLoader(self.test_data, batch_size=64, shuffle=False)
    
    def get_train_data(self):
        return self.train_data

    def get_test_data(self):
        return self.test_data
    
    def visualize_dataset(self):
        import matplotlib.pyplot as plt
        import numpy as np

        # Iterator and first batch of data/lables
        data_iter = iter(self.train_data_loader)
        images, labels = next(data_iter)

        fig = plt.figure(figsize=(25, 4))
        for idx in np.arange(20):
            # subplot 2 rows and 10 columns, no grid, index starting from 1
            ax = fig.add_subplot(2, 10, idx+1, xticks=[], yticks=[])
            # Reshape the image (2D) & make it grayscale
            ax.imshow(np.squeeze(images[idx]), cmap='gray')
            # Print the label as title
            ax.set_title(str(labels[idx].item()))
        plt.show()

ds = Dataset('data')
ds.visualize_dataset()