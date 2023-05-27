from random import *
import torch.nn as nn
import torch
import math
from config import *

a = torch.rand(2,1,768)

a = torch.Tensor([[[1,2,3],[4,5,6],[7,8,9]],[[10,11,12],[13,14,15],[16,17,18]]])

index = torch.LongTensor([[[1,1,1],[2,2,2]],[[0,0,0],[1,1,1]]])

c = torch.gather(a,dim=1,index=index)
print(c)

