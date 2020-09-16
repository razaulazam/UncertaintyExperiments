import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet5(nn.Module):
    def __init__(self, output_dim, layer_drop=0.5):
        super().__init__()
        self.output_dim = output_dim*2
        self.num_classes = output_dim
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 =  nn.BatchNorm2d(128)

        self.max_pool1 = nn.MaxPool2d(kernel_size=2)
        self.max_pool2 = nn.MaxPool2d(kernel_size=2)
        self.activation = nn.ReLU()

        self.fc_1 = nn.Linear(in_features=2048, out_features=1024)
        self.fc_2 = nn.Linear(in_features=1024, out_features=512)
        self.fc_3 = nn.Linear(in_features=512, out_features=self.output_dim)
        self.drop_layer = nn.Dropout(p=layer_drop)
        

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.max_pool1(x)
        x = self.activation(x)
        
        x = self.drop_layer(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.activation(x)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.max_pool2(x)
        x = self.activation(x)

        x = x.view(x.shape[0], -1)
        
        x = self.fc_1(x)
        x = self.activation(x)
        x = self.drop_layer(x)

        x = self.fc_2(x)
        x = self.activation(x)

        x = self.fc_3(x)
        
        x = torch.split(x, self.num_classes, dim=1)
        logits = x[0]
        variances = F.softplus(x[1])
        x = [logits, variances]
        x = torch.cat([i.unsqueeze(1) for i in x], 1)

        return x


