import torch, torchvision
from torchvision import datasets, transforms    
import matplotlib.pyplot as plt

# MNIST example from paper is in binarized mnist
class Binarize(object):
    def __call__(self, tensor):
        return (tensor > 0.5).float()

class Dataset():

    def __init__(self, data_dir):
        self.data_dir = data_dir
        # Binirize MNIST
        self.transform = transforms.Compose([transforms.ToTensor(),
                              Binarize(),
                              torch.nn.Flatten(0)])
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
        import itertools
        images = torch.stack(tuple(zip(*tuple(itertools.islice(iter(self.test_data), 16))))[0])  # You don't need to understand this code...
        plt.imshow(1 - torchvision.utils.make_grid(images.unflatten(1, (1, 28, 28))).permute(1, 2, 0))
        plt.title('Your model should predict the correct number for each image.')
        plt.axis('off')

    


ds = Dataset('data')
ds.visualize_dataset()