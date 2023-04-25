import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR

from app.utils.utils import get_val_loss
from app.utils.logging import TqdmToLogger, Logger
from tqdm import tqdm
import numpy as np

import copy
import logging
import os


log = logging.getLogger(__name__)
log.setLevel(logging.INFO)






def train(model, epochs, loss_function, optimizer, scheduler, data, checkpoint_path, device):
    
    min_epochs = 1
    best_model = None
    min_val_loss = 5

    if not os.path.isdir(f'checkpoints'):
        os.mkdir(f'checkpoints')


    logger = Logger(name='app/train', isTqdm=True, filename='outputs/process.log')
    tqdm_out = TqdmToLogger(logger, level=logging.INFO)
    
    
    
    for epoch in tqdm(range(epochs), bar_format='{l_bar}{bar:60}{r_bar}{bar:-60b}', file=tqdm_out):
        train_loss = []
        for (seq, label) in data['train_data']:
            seq = seq.to(device)
            label = label.to(device)
            y_pred = model(seq)
            loss = loss_function(y_pred, label)
            train_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        scheduler.step()

        val_loss = get_val_loss(model, data['val_data'], device)
        if epoch + 1 >= min_epochs and val_loss < min_val_loss:
            min_val_loss = val_loss
            best_model = copy.deepcopy(model)
            state = {'models': model.state_dict()}
            torch.save(state, f'outputs/checkpoints/{epoch}.pkl')

        logger.info('epoch {:03d} train_loss {:.8f} val_loss {:.8f}'.format(epoch, np.mean(train_loss), val_loss))
        model.train()
        print(f'loss: {loss}')
        print(f'val_loss: {val_loss}')

    if epoch > min_epochs and min_val_loss != 5:
        state = {'models': best_model.state_dict()}
        torch.save(state, f'outputs/checkpoints/best.pkl')








