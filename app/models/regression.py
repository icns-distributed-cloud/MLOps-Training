import torch
import torch.nn as nn



class Regression(nn.Module):
    def __init__(self, input_size, output_size):
        super(Regression, self).__init__()
        self.linear = nn.Linear(in_features=input_size, out_features=output_size)


    def forward(self, x):
        x_ = self.linear(x)
        return x_