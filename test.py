from random import *
import torch.nn as nn
import torch
import math
from config import *
from torch.optim import optimizer

a = torch.rand(1,5,46)
b = a.data.max(2)[1]
print(b)