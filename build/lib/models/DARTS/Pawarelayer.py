import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable
import torch.nn.functional as F


if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

class Partial_aware_layer_or(nn.Module):
    """ Partial aware layer for special re-id search space
    """
    def __init__(self, C_in, C_out, stride, body_part =4, sigma = 1, affine=True):
        super().__init__()
        self.body_part = body_part 
        self.sigma = 1
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1,1))
        self.linear_transfer = nn.Linear(C_in, C_in)
        self.soft_attention = nn.Linear(C_in, C_in)
        self.fusion_conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(C_in*2, C_out, 1, stride, 0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )
        

    def forward(self, x):
        bs, c, hk, wk = x.size()
        #print(bs, c, hk, wk)
        seg_height = hk // self.body_part
        #offset = hk % self.body_part  后来发现offeset 可以不用
        enhance_metrix = torch.FloatTensor(bs, c, hk, wk).to(device)
        #print(enhance_metrix)
        for i in range(self.body_part):
            out_vector = self.adaptive_pool(x[:,:,i*seg_height:(i+1)*seg_height,:])
            attention = self.soft_attention(torch.squeeze(out_vector))
            transform = self.linear_transfer(torch.squeeze(out_vector))
            enhanced_vector = F.softmax(attention*transform, dim=1) * torch.squeeze(out_vector) *self.sigma
            enhance_metrix[:,:,i*seg_height:(i+1)*seg_height,:] = enhanced_vector.unsqueeze(2).unsqueeze(3).repeat(1,1,seg_height,wk)
        
        outfeature = self.fusion_conv(torch.cat((x, enhance_metrix), 1))
        return outfeature

        

class Partial_aware_layer_improved(nn.Module):
    """ Partial aware layer for special re-id search space
    """
    """ Another enhance matrix
    """
    def __init__(self, C_in, C_out, stride, body_part =4, sigma = 1, affine=True):
        super().__init__()
        self.body_part = body_part 
        self.sigma = 1
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1,1))
        self.linear_transfer = nn.Linear(C_in, C_in)
        self.soft_attention = nn.Linear(C_in, C_in)
        self.fusion_conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(C_in*2, C_out, 1, stride, 0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )
        

    def forward(self, x):
        bs, c, hk, wk = x.size()
        seg_height = hk // self.body_part
        #offset = hk % self.body_part 
        enhance_metrix = torch.FloatTensor(bs, c, hk, wk).to(device)
        for i in range(self.body_part):
            out_vector = self.adaptive_pool(x[:,:,i*seg_height:(i+1)*seg_height,:])
            attention = self.soft_attention(torch.squeeze(out_vector))
            transform = self.linear_transfer(torch.squeeze(out_vector))
            enhanced_vector = F.softmax(attention*transform, dim=1) * torch.squeeze(out_vector) *self.sigma
            enhance_metrix[:,:,i*seg_height:(i+1)*seg_height,:] = enhanced_vector.unsqueeze(2).unsqueeze(3)*x[:,:,i*seg_height:(i+1)*seg_height,:]
        
        outfeature = self.fusion_conv(torch.cat((x, enhance_metrix), 1))
        return outfeature

if __name__ == "__main__":
    PAL = Partial_aware_layer_improved(32, 32, 0)
    PAL.to(device)
    x = torch.randn(2, 32, 256, 128)
    out = PAL(Variable(x).to(device))
    print(out.size()) 

    PAL = Partial_aware_layer_or(32, 32, 0)
    PAL.to(device)
    x = torch.randn(2, 32, 256, 128)
    out = PAL(Variable(x).to(device))
    print(out.size()) 

    

        
        