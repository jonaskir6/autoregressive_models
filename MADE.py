import torch
from torch import nn




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
