from app.utils.config import load_config
from app.utils.logging import Logger
from app.ml_engine import MLEngine

import json
import argparse


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, help='config')
    args = parser.parse_args()

    return args



if __name__ == '__main__':
    '''
    args = arg_parser()

    config = json.loads(args.config)

    # config = load_config('./configs/lstm_train.json')
    logger = Logger('app')

    
    ME = MLEngine(config['model'])
    ME.run()
    
   '''
    args = arg_parser()

    # config = json.loads(args.config)
    config = load_config('./configs/lstm_train.json')
    
    logger = Logger('app')

    ME = MLEngine(config['model'])
    ME.run()
    