import torch
from torch import nn


class MADE(nn.Module):
    def __init__(self, input_feat, num_units, num_layer, ordering):
        super(MADE,self).__init__()
        
        self.m_k=[]
        self.m_k.append(ordering)
        D=input_feat-1
        for j in range(num_layer-1):
            self.m_k.append(torch.randint(low=min(self.m_k[j-1]),high=D,size=(num_units,)))
        

        layers=[]
        layers.append(MaskedLayer(in_feat=input_feat,out_feat=num_units,m_k=self.m_k[1],m_k_prev=self.m_k[0],layer_type="hidden"))
        layers.append(nn.ReLU())
        for i in range(2,num_layer):
            layers.append(MaskedLayer(num_units,num_units,m_k=self.m_k[i],m_k_prev=self.m_k[i-1],layer_type="hidden"))
            layers.append(nn.ReLU())


        layers.append(MaskedLayer(num_units,input_feat,m_k=ordering,m_k_prev=self.m_k[num_layer-1],layer_type="output"))
        layers.append(nn.Sigmoid())


        self.layer=nn.ModuleList(layers)

    def shuffle_ordering(self, ordering):
        self.layer[0].update_mask(self, m_k = self.m_k[1], m_k_prev = ordering)

    def forward(self,input):
        x = input
        for layer in self.layer:
            x = layer(x)
        return x
    

class MaskedLayer(nn.Linear):
    def __init__(self, in_feat, out_feat, m_k, m_k_prev, layer_type):
        super().__init__(in_features=in_feat, out_features=out_feat)
        self.register_buffer('mask', torch.zeros(out_feat, in_feat))

        for j in range(in_feat):
            for i in range(out_feat):
                if layer_type=="output":
                    if m_k[i]>m_k_prev[j]:
                        self.mask[i,j]=1
                else:
                    if m_k[i]>=m_k_prev[j]:
                        self.mask[i,j]=1
    
    def update_mask(self, m_k, m_k_prev):
        self.mask.fill_(0)
        for j in range(self.in_features):
            for i in range(self.out_features):
                if m_k[i]>=m_k_prev[j]:
                    self.mask[i,j]=1
         

    def forward(self, x):
        return nn.functional.linear(x, self.weight * self.mask, self.bias)
