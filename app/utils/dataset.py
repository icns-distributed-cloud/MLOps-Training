from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import torch

import os
import json
import pandas as pd
import numpy as np


# todo: separate train, val, test data loader

def load_csv_data(filename):
    # path = os.path.dirname(os.path.realpath(__file__)) + f'/../data/{filename}'
    filename = filename[9:]
    print(os.listdir())
    path = f'data/{filename}'
    
    try:
        df = pd.read_csv(path, encoding='gbk')
    except:
        df = pd.read_csv(path, encoding='utf-8')
    df.fillna(df.mean(), inplace=True)
    return df




class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)



def load_data(filename, seq_len, batch_size, flag, output_size, input_columns, output_columns):
    if flag == 'regression':
        Dtr, Dte = load_regression_data(filename)
        pass

    
    # unvariate singlestep
    elif flag == 'us':
        Dtr, Val, Dte, min_max_dict = nn_seq_us(filename, B=batch_size, input_columns=input_columns, output_columns=output_columns, seq_len=seq_len)
    
    
    # multivariate singlestep
    elif flag == 'ms':
        Dtr, Val, Dte, min_max_dict = nn_seq_ms(filename, B=batch_size, input_columns=input_columns, output_columns=output_columns, seq_len=seq_len)
    
    
    # multivariate multistep
    elif flag == 'mm':
        Dtr, Val, Dte, min_max_dict = nn_seq_mm(filename, B=batch_size, num=12, input_columns=input_columns, output_columns=output_columns, seq_len=seq_len)
    
    
    elif flag == 'test':
        Dtr, Val, Dte, min_max_dict = test(filename, B=batch_size, seq_len=seq_len)


    with open('outputs/min_max.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(min_max_dict, ensure_ascii=False)) 
    # return Dtr, Val, Dte, m, n

    return dict(
        train_data=Dtr,
        val_data=Val,
        test_data=Dte,
        min_max_dict=min_max_dict
    )


def load_regression_data(filename):
    print('data processing...')
    dataset = load_csv_data(filename)
    


def test(filename, B, seq_len):
    print('data processing... test')

    dataset = load_csv_data(filename)
    a = dataset.iloc[:, [1,2]]
    print(a)

    
    return 1,2,3,4,5





def nn_seq_mm(filename, B, num, input_columns, output_columns, seq_len=1):
    print('data processing...')
    dataset = load_csv_data(filename)
    # split
    train = dataset[:int(len(dataset) * 0.6)]
    val = dataset[int(len(dataset) * 0.6):int(len(dataset) * 0.8)]
    test = dataset[:int(len(dataset) * 0.2)]

    # test = dataset[int(len(dataset) * 0.8):len(dataset)]

    min_max_dict = {}
    for input_column in input_columns:
        column_idx = dataset.columns.get_loc(input_column)
        m, n = np.max(train[input_column]), np.min(train[input_column])
        min_max_dict[input_column] = {'idx': column_idx, 'max': m, 'min': n}

    output_column_idx = dataset.columns.get_loc(output_columns[0])
    m, n = np.max(train[output_columns[0]]), np.min(train[output_columns[0]])
    min_max_dict['output'] = {'idx': output_column_idx, 'max': m, 'min': n}

    # m, n = np.max(train[train.columns[1]]), np.min(train[train.columns[1]])
    print(min_max_dict)
    
    def process(data, batch_size, step_size):
        scaler = MinMaxScaler()
        data[input_columns] = scaler.fit_transform(data[input_columns])

        output_column_data = data[output_columns[0]]
        output_column_data = output_column_data.tolist()
        data = data.values.tolist()

        seq = []
        for i in range(0, len(data) - seq_len - num, step_size):
            train_seq = []
            train_label = []

            for j in range(i, i + seq_len):
                x = []
                for input_column in input_columns:
                    x.append(data[j][min_max_dict[input_column]['idx']])
                train_seq.append(x)
            for j in range(i + seq_len, i + seq_len + num):
                train_label.append(output_column_data[j])

            
            train_seq = torch.FloatTensor(train_seq)
            train_label = torch.FloatTensor(train_label).view(-1)
            seq.append((train_seq, train_label))





        '''
        load = data[data.columns[1]]
        data = data.values.tolist()
        load = (load - n) / (m - n)
        load = load.tolist()
        seq = []
        for i in range(0, len(data) - seq_len - num, step_size):
            train_seq = []
            train_label = []

            for j in range(i, i + seq_len):
                x = [load[j]]
                for c in range(2, 8):
                    x.append(data[j][c])
                train_seq.append(x)
            for j in range(i + seq_len, i + seq_len + num):
                train_label.append(load[j])

            
            train_seq = torch.FloatTensor(train_seq)
            train_label = torch.FloatTensor(train_label).view(-1)
            seq.append((train_seq, train_label))

        '''

        # print(seq[-1])
        seq = MyDataset(seq)
        seq = DataLoader(dataset=seq, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)

        return seq

    Dtr = process(train, B, step_size=1)
    Val = process(val, B, step_size=1)
    Dte = process(test, B, step_size=num)

    return Dtr, Val, Dte, min_max_dict
 

# Multivariate-SingleStep-LSTM data processing.
def nn_seq_ms(filename, B, input_columns, output_columns, seq_len=1):
    print('data processing...')

    dataset = load_csv_data(filename)
    # split
     
    
    train = dataset[:int(len(dataset) * 0.6)]
    val = dataset[int(len(dataset) * 0.6):int(len(dataset) * 0.8)]
    
    test = dataset[:int(len(dataset) * 0.2)]
    # test = dataset[int(len(dataset) * 0.8):len(dataset)]    
    
    min_max_dict = {}
    for input_column in input_columns:
        column_idx = dataset.columns.get_loc(input_column)
        m, n = np.max(train[input_column]), np.min(train[input_column])
        # m, n = np.max(train[train.columns[column_idx]]), np.min(train[train.columns[column_idx]])
        min_max_dict[input_column] = {'idx': column_idx, 'max': m, 'min': n}

     
    output_column_idx = dataset.columns.get_loc(output_columns[0])
    m, n = np.max(train[output_columns[0]]), np.min(train[output_columns[0]])
    # m, n = np.max(train[train.columns[output_column_idx]]), np.min(train[train.columns[output_column_idx]])
    min_max_dict['output'] = {'idx': output_column_idx, 'max': m, 'min': n}
    
    # m, n = np.max(train[train.columns[1]]), np.min(train[train.columns[1]])
    print(min_max_dict)
    def process(data, batch_size, input_columns, output_columns):
        scaler = MinMaxScaler()
        data[input_columns] = scaler.fit_transform(data[input_columns])

        output_column_data = data[output_columns[0]]
        output_column_data = output_column_data.tolist()
        data = data.values.tolist()

        seq = []
        for i in range(len(data) - seq_len):
            train_seq = []
            train_label = []
            for j in range(i, i + seq_len):
                x = []
                for input_column in input_columns:
                    x.append(data[j][min_max_dict[input_column]['idx']])

                train_seq.append(x)

            train_label.append(output_column_data[i + seq_len])
            train_seq = torch.FloatTensor(train_seq)
            train_label = torch.FloatTensor(train_label).view(-1)
            seq.append((train_seq, train_label))


        '''
        input_columns_idx_list = []

        scaled_input_data = {}
        for input_column in input_columns:
            # column_data = data.columns[data.columns.get_loc(input_column)]
            # column_data = (data[column_data] - min_max_dict[input_column]['min']) / (min_max_dict[input_column]['max'] - min_max_dict[input_column]['min'])
            input_column_data = (data[input_column] - min_max_dict[input_column]['min']) / (min_max_dict[input_column]['max'] - min_max_dict[input_column]['min'])
            input_column_data = input_column_data.tolist()

            scaled_input_data[input_column] = input_column_data

            # input_columns_idx_list.append(data.columns.get_loc(input_column))

        # output_column_data = data.columns[data.columns.get_loc(output_columns[0])]
        # scaled_output_data = (data[output_column_data] - min_max_dict[output_columns[0]]['min']) / (min_max_dict[output_columns[0]]['max'] - min_max_dict[output_columns[0]]['min'])
        
        output_column_data = (data[output_columns[0]] - min_max_dict[output_columns[0]]['min']) / (min_max_dict[output_columns[0]]['max'] - min_max_dict[output_columns[0]]['min'])
        scaled_output_data = output_column_data.tolist()

        seq = []
        for i in range(len(data) - seq_len):
            train_seq = []
            train_label = []
            for j in range(i, i + seq_len):
                x = []
                for input_column in input_columns:
                    x.append(scaled_input_data[input_column])
                
                train_seq.append(x)
            
            train_label.append(scaled_output_data[i + seq_len])
            train_seq = torch.FloatTensor(train_seq)
            train_label = torch.FloatTensor(train_label).view(-1)
            seq.append((train_seq, train_label))
        '''
            


        '''
        load = data[data.columns[output_column_idx]]
        data = data.values.tolist()
        load = (load - n) / (m - n)
        load = load.tolist()
        seq = []


        for i in range(len(data) - seq_len):
            train_seq = []
            train_label = []
            for j in range(i, i + seq_len):
                # x = [load[j]]
                x = []
                for c in input_columns_idx_list:
                    x.append(data[j][c])

                train_seq.append(x)
            train_label.append(load[i + seq_len])
            train_seq = torch.FloatTensor(train_seq)
            train_label = torch.FloatTensor(train_label).view(-1)
            seq.append((train_seq, train_label))
        '''

        # print(seq[-1])
        seq = MyDataset(seq)
        seq = DataLoader(dataset=seq, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)

        return seq

    Dtr = process(train, B, input_columns, output_columns)
    Val = process(val, B, input_columns, output_columns)
    Dte = process(test, B, input_columns, output_columns)

    
    return Dtr, Val, Dte, min_max_dict


# Univariate-SingleStep-LSTM data processing.
def nn_seq_us(filename, B, input_columns, output_columns, seq_len=1):
    print('data processing...')
    dataset = load_csv_data(filename)
    # split
    train = dataset[:int(len(dataset) * 0.6)]
    val = dataset[int(len(dataset) * 0.6):int(len(dataset) * 0.8)]
    test = dataset[int(len(dataset) * 0.8):len(dataset)]
    m, n = np.max(train[train.columns[1]]), np.min(train[train.columns[1]])

    def process(data, batch_size):
        load = data[data.columns[1]]
        data = data.values.tolist()
        load = (load - n) / (m - n)
        load = load.tolist()
        seq = []
        for i in range(len(data) - seq_len):
            train_seq = []
            train_label = []
            for j in range(i, i + seq_len):
                x = [load[j]]
                train_seq.append(x)
            # for c in range(2, 8):
            #     train_seq.append(data[i + 24][c])
            train_label.append(load[i + seq_len])
            train_seq = torch.FloatTensor(train_seq)
            train_label = torch.FloatTensor(train_label).view(-1)
            seq.append((train_seq, train_label))

        # print(seq[-1])
        seq = MyDataset(seq)
        seq = DataLoader(dataset=seq, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)

        return seq

    Dtr = process(train, B)
    Val = process(val, B)
    Dte = process(test, B)

    return Dtr, Val, Dte, m, n