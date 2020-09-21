import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from deepLabv3 import DeepLab
from pprint import pprint


class testmodel(nn.Module):
    def __init__(self): 
        super(testmodel, self).__init__()
        self.l1 = nn.Linear(10, 20)
        self.l2 = nn.Linear(20, 100)
        self.l3 = nn.Dropout(p=0.2)
        self.l4 = nn.Linear(100, 4)
        
    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        
        return x


model = testmodel()
model_deep = DeepLab()
pprint(model_deep)

    

    

