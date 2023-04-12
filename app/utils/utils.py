import numpy as np

import torch
from torch import nn



def get_mape(x, y):
    return np.mean(np.abs((x - y) / x))


def get_val_loss(model, Val, device):
    model.eval()
    loss_function = nn.MSELoss().to(device)
    val_loss = []
    for (seq, label) in Val:
        with torch.no_grad():
            seq = seq.to(device)
            label = label.to(device)
            y_pred = model(seq)
            loss = loss_function(y_pred, label)
            val_loss.append(loss.item())

    return np.mean(val_loss)