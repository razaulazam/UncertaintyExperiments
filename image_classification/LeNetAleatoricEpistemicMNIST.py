import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet5(nn.Module):
    def __init__(self, output_dim, layer_drop=0.5):
        super().__init__()
        self.output_dim = output_dim*2
        self.num_classes = output_dim
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)

        self.max_pool = nn.MaxPool2d(kernel_size=2)
        self.activation = nn.ReLU()

        self.fc_1 = nn.Linear(in_features=256, out_features=120)
        self.fc_2 = nn.Linear(in_features=120, out_features=84)
        self.fc_3 = nn.Linear(in_features=84, out_features=self.output_dim)
        self.drop_layer = nn.Dropout(p=layer_drop)

    def forward(self, x):
        x = self.conv1(x)
        x = self.max_pool(x)

        x = self.activation(x)

        x = self.conv2(x)
        x = self.max_pool(x)

        x = self.activation(x)
        
        x = self.drop_layer(x)  # 'DropOut Layer for capturing Epistemic Uncertainty'

        x = x.view(x.shape[0], -1)
        x = self.fc_1(x)
        x = self.activation(x)
        x = self.drop_layer(x)  # 'DropOut Layer for capturing Epistemic Uncertainty'


        x = self.fc_2(x)

        x = self.activation(x)

        x = self.fc_3(x)
        x = torch.split(x, self.num_classes, dim=1)
        logits = x[0]
        variances = F.softplus(x[1])
        x = [logits, variances]
        x = torch.cat([i.unsqueeze(1) for i in x], 1)

        return x


