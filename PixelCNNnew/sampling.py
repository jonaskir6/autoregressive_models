import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

def sampling(model, device):

  channels = 3
  img_size = 32

  sample = torch.Tensor(1, channels, img_size, img_size).to(device)
  sample.fill_(0.)
  colors = ['r', 'g', 'b']

  for c in range(channels):  
    for i in range(img_size):
      for j in range(img_size):
          out = model(sample, colors[c])
          probs = F.softmax(out[:,:,i,j], dim=-1).data
          sample[:,c,i,j] = torch.multinomial(probs, 1).float() / 255.0

  plt.imshow(sample[0].permute(1, 2, 0).to('cpu'))
  plt.show()

def samplingmnist(model, device):
  num_imgs = 5
  img_chn = 1
  img_size = 28

  sample = torch.Tensor(num_imgs, img_chn, img_size, img_size).to(device)
  sample.fill_(0.)

  for i in range(img_size):
    for j in range(img_size):
      out = model(sample)
      probs = F.softmax(out[:,:,i,j], dim=-1).data
      sample[:,:,i,j] = torch.multinomial(probs, 1).float() / 255.0
      
  for idx in range (num_imgs):
    plt.imshow(sample[idx].permute(1, 2, 0).to('cpu'), cmap='Greys_r')
    plt.show()