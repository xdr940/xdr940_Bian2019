import  torch

import torch.nn.functional as F
w=3
h=4
t = torch.linspace(start=0,steps=w*h,end=w*h-1).reshape([1,1,w,h])
print(t)
b,c,h,w = t.shape

flow = torch.ones([b,h,w,2])
flow[:,:,:,0]=-1
#flow = torch.tensor([0,0,0,0,
#                      1,1,1,1,
##                      1,1,1,1.,
 #                    0,0,0,0,
 #                     1,1,1,1,
 #                     2,2,2,2]).reshape([1,3,4,2])
print(flow)
b = F.grid_sample(input=t,mode='bilinear',grid=flow)
#print(b.shape)
print(b)