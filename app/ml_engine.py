from app.utils.dataset import load_data
from app.train import train
from app.test import test

import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR

import logging.config

    


log = logging.getLogger(__name__)


class MLEngine:
    def __init__(self, model_config):
        self.model_config = model_config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        
        self.data = {}

        self.__parse_config()
        self.__load_dataset()

    
    def __parse_config(self):
        self.type = self.model_config['type']

        if self.model_config['model_type'] == 'LSTM':
            from app.models.lstm import Net
            self.model = Net(input_size=len(self.model_config['input_columns']),
                                hidden_size=self.model_config['hidden_size'],
                                output_size=len(self.model_config['output_columns'])).to(self.device)

        elif self.model_config['model_type'] == 'Regression':
            from app.models.regression import Regression
            self.model = Regression(input_size=len(self.model_config['input_columns']),
                                    output_size=len(self.model_config['output_columns']))

        else:
            # todo: other model type
            pass 
        
        if self.type == 'train':
            self.__load_loss_function()
            self.__load_optimizer()
            self.__load_scheduler()

        elif self.type == 'test':
            pass

        



    def __load_dataset(self):
        self.data = load_data(filename=self.model_config['dataset_path'],
                                seq_len=self.model_config['seq_len'], 
                                batch_size=self.model_config['batch_size'],
                                flag=self.model_config['flag'], 
                                output_size=len(self.model_config['output_columns']),
                                input_columns=self.model_config['input_columns'],
                                output_columns=self.model_config['output_columns'])




    
    def run(self):
        train(model=self.model,
                epochs=self.model_config['epochs'],
                loss_function=self.loss_function,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                data=self.data,
                checkpoint_path=f'outputs/checkpoints',
                device=self.device)
    
        test(model=self.model,
                data=self.data,
                checkpoint_path=f'outputs/checkpoints',
                # state_dict_name=self.model_config['state_dict_name'],
                device=self.device)

        '''
        if self.type == 'train':
            train(model=self.model,
                    epochs=self.model_config['epochs'],
                    loss_function=self.loss_function,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    data=self.data,
                    checkpoint_path=self.model_config['checkpoint_path'],
                    device=self.device)

            pass

        elif self.type == 'test':
            test(model=self.model,
                    data=self.data,
                    checkpoint_path=self.model_config['checkpoint_path'],
                    state_dict_name=self.model_config['state_dict_name'],
                    device=self.device)

            pass
        '''


    def __load_loss_function(self):
        if self.model_config['loss'] == 'MSE':
            self.loss_function = nn.MSELoss().to(self.device)

        else:
            # todo: other loss functions
            pass


    def __load_optimizer(self):
        if self.model_config['optimizer'] == 'Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), 
                                        lr=self.model_config['lr'], 
                                        weight_decay=self.model_config['weight_decay'])
        
        elif self.model_config['optimizer'] == 'SGD':
            self.optimizer = torch.optim.SGD(self.model.parameters(), 
                                        lr=self.model_config['lr'], 
                                        momentum=0.9, 
                                        weight_decay=self.model_config['weight_decay'])
    
        else:
            # todo: other optimizers
            pass

    def __load_scheduler(self):
        self.scheduler = StepLR(self.optimizer, 
                                step_size=self.model_config['step_size'], 
                                gamma=self.model_config['gamma'])