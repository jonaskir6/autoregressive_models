import torch
from torchvision import datasets, transforms    
import ssl


ssl._create_default_https_context = ssl._create_unverified_context

class Dataset():

    def __init__(self, dataset, batch_size=64, data_dir='data'):
        self.dataset = dataset
        if dataset=='cifar10':
            self.transform = transforms.Compose([
                transforms.ToTensor(),
            ])
        elif dataset=='mnist':
            self.transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=3),
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
            ])
        # Train and test data
        if dataset == 'mnist':
            self.train_data = datasets.MNIST(data_dir, train=True, download=True, transform=self.transform)
            self.test_data = datasets.MNIST(data_dir, train=False, transform=self.transform)
        elif dataset == 'cifar10':
            self.train_data = datasets.CIFAR10(data_dir, train=True, download=True, transform=self.transform)
            self.test_data = datasets.CIFAR10(data_dir, train=False, transform=self.transform)
        else:
            raise ValueError('Dataset not implemented')
        # Data loaders
        self.train_data_loader = torch.utils.data.DataLoader(self.train_data, batch_size, shuffle=True, drop_last=True)
        self.test_data_loader = torch.utils.data.DataLoader(self.test_data, batch_size, shuffle=False, drop_last=True)
    
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

        # Define the CIFAR-10 class names
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

        # Iterator and first batch of data/labels
        data_iter = iter(data_loader)
        images, labels = next(data_iter)


        if self.dataset=='cifar10':
            fig = plt.figure(figsize=(25, 8))
            for idx in np.arange(40): 
                ax = fig.add_subplot(4, 10, idx+1, xticks=[], yticks=[]) 
                if images.shape[1] == 1:
                    ax.imshow(np.squeeze(images[idx]))
                else:
                    ax.imshow(np.transpose(images[idx], (1, 2, 0)))
                ax.set_title(class_names[labels[idx].item()], fontsize=12)

                plt.subplots_adjust(hspace=0.5)  
            
        elif self.dataset=='mnist':
            fig = plt.figure(figsize=(25, 4))
            for idx in np.arange(20):
                # subplot 2 rows and 10 columns, no grid, index starting from 1
                ax = fig.add_subplot(2, 10, idx+1, xticks=[], yticks=[])
                # Reshape the image (2D) & make it grayscale
                ax.imshow(np.squeeze(images[idx]).view(28, 28), cmap='gray')
                # Print the label as title
                ax.set_title(str(labels[idx].item()))
        plt.show()