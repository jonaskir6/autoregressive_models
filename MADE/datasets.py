import torch
from torchvision import datasets, transforms    

# MNIST example from paper is in binarized mnist
class Binarize():
    def __call__(self, tensor):
        return (tensor > 0.5).float()

class Dataset():

    def __init__(self, dataset, batch_size=64, data_dir='data'):
        self.data_dir = data_dir
        # Binarize MNIST
        self.transform = transforms.Compose([transforms.ToTensor(),
                              Binarize(),
                              torch.nn.Flatten(0)])
        # Train and test data
        if dataset == 'mnist':
            self.train_data = datasets.MNIST(self.data_dir, train=True, download=True, transform=self.transform)
            self.test_data = datasets.MNIST(self.data_dir, train=False, transform=self.transform)
        else:
            raise ValueError('Dataset not implemented')
        # Data loaders
        self.train_data_loader = torch.utils.data.DataLoader(self.train_data, batch_size, shuffle=True)
        self.test_data_loader = torch.utils.data.DataLoader(self.test_data, batch_size, shuffle=False)
    
    def get_train_data_loader(self):
        return self.train_data_loader

    def get_test_data_loader(self):
        return self.test_data_loader
    
    def visualize_dataset(self, data_loader):
        """
        Visualize the dataset
        Args:
            data_loader: train_data_loader or test_data_loader (train or test dataset)
        """
        import matplotlib.pyplot as plt
        import numpy as np

        # Iterator and first batch of data/lables
        data_iter = iter(data_loader)
        images, labels = next(data_iter)

        fig = plt.figure(figsize=(25, 4))
        for idx in np.arange(20):
            # subplot 2 rows and 10 columns, no grid, index starting from 1
            ax = fig.add_subplot(2, 10, idx+1, xticks=[], yticks=[])
            # Reshape the image (2D) & make it grayscale
            ax.imshow(np.squeeze(images[idx]).view(28, 28), cmap='gray')
            # Print the label as title
            ax.set_title(str(labels[idx].item()))
        plt.show()
        