import torch

from app.utils.utils import  get_mape
from app.models.lstm import  Net
from tqdm import tqdm
from itertools import chain
from scipy.interpolate import make_interp_spline

import matplotlib.pyplot as plt
import numpy as np

import argparse
import logging



log = logging.getLogger(__name__)




def test(model, data, checkpoint_path, state_dict_name=None, device='cpu'):
    model.load_state_dict(torch.load(f'outputs/checkpoints/best.pkl')['models'])
    model.eval()

    pred = []
    y = []

    for (seq, target) in data['test_data']:
        target = list(chain.from_iterable(target.data.tolist()))
        y.extend(target)
        seq = seq.to(device)
        with torch.no_grad():
            y_pred = model(seq)
            y_pred = list(chain.from_iterable(y_pred.data.tolist()))
            pred.extend(y_pred)

    y, pred = np.array(y), np.array(pred)

    m = data['min_max_dict']['output']['max']
    n = data['min_max_dict']['output']['min']
    print(f'm: {m}')
    print(f'n: {n}')

    y = (m - n) * y + n
    pred = (m - n) * pred + n
    print(f'y: {y}')
    print(f'pred: {pred}')


    # print('mape:', get_mape(y, pred))

    # plot
    x = [i for i in range(1, 151)]
    x_smooth = np.linspace(np.min(x), np.max(x), 900)
    y_actual = make_interp_spline(x, y[150:300])(x_smooth)
    plt.plot(x_smooth, y_actual, c='green', marker='*', ms=1, alpha=0.75, label='true')

    y_predicted = make_interp_spline(x, pred[150:300])(x_smooth)
    plt.plot(x_smooth, y_predicted, c='red', marker='o', ms=1, alpha=0.75, label='pred')
    plt.grid(axis='y')
    plt.legend()
    plt.savefig('outputs/fig1.png', dpi=300)

    plt.clf()

    plt.scatter(y_actual, y_predicted, alpha=0.4)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('MULTIPLE LINEAR REGRESSION')
    plt.savefig('outputs/fig2.png', dpi=300)
    
    
  
    

