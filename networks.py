import torch
from torch import nn


class MADE(nn.Module):
    def __init__(self, input_feat, num_units, num_layer, ordering):
        super(MADE,self).__init__()
        
        self.m_k=[]
        self.m_k.append(ordering)
        D=input_feat-1
        for j in range(num_layer):
            self.m_k.append(torch.randint(low=min(self.m_k[j-1]),high=D,size=num_units))


        layers=[]
        layers.append(MaskedLayer(input_feat,num_units,m_k=self.m_k[1],m_k_prev=self.m_k[0]))
        layers.append(nn.BatchNorm1d(num_units))
        layers.append(nn.ReLU())
        for i in range(2,num_layer-1):
            layers.append(MaskedLayer(num_units,num_units,self.m_k[i],self.m_k[i-1]))
            layers.append(nn.BatchNorm1d(num_units))
            layers.append(nn.ReLU)


        layers.append(MaskedLayer(num_units,input_feat,self.m_k[num_layer],self.m_k[num_layer-1],"output"))


        self.layer=nn.Sequential(*layers)

    def forward(self,input):
        return self.layer(input)


class MaskedLayer(nn.Linear):

    def __init__(self, in_feat, out_feat, m_k, m_k_prev, layer_type):
        super().__init__(in_features=in_feat, out_features=out_feat)
        self.mask= torch.zeros(out_feat, in_feat)

        for j in range(in_feat):
            for i in range(out_feat):
                if layer_type=="output":
                    if m_k[j]>m_k_prev[i]:
                        self.mask[i,j]=1
                else:
                    if m_k[j]>=m_k_prev[i]:
                        self.mask[i,j]=1
            

    def forward(self, x):
        return nn.functional.linear(x, self.weight * self.mask, self.bias)
