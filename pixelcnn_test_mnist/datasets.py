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
            self.transform = transforms.ToTensor()
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
        self.train_data_loader = torch.utils.data.DataLoader(self.train_data, batch_size, shuffle=True)
        self.test_data_loader = torch.utils.data.DataLoader(self.test_data, batch_size, shuffle=False)
    
    def get_train_data_loader(self):
        return self.train_data_loader

    def get_test_data_loader(self):
        return self.test_data_loader
    