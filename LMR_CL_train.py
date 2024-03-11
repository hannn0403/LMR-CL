import os
import pickle
import numpy as np
from random import random

from config import get_config, activation_dict
from data_loader_only_bert_feature_20 import get_loader
from solver_only_bert import Solver

import torch
import torch.nn as nn
from torch.nn import functional as F
 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
if __name__ == '__main__':

    # Setting random seed
    
    random_name = str(random())
    random_seed = 336   
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 
    np.random.seed(random_seed)
    
    # Setting the config for each stage
    train_config = get_config(mode='train')
    test_config = get_config(mode='test')

    print(train_config)

    # Creating pytorch dataloaders
    train_data_loader = get_loader(train_config, shuffle = True)
    test_data_loader = get_loader(test_config, shuffle = False)


    # Solver is a wrapper for model traiing and testing
    solver = Solver
    solver = solver(train_config, test_config, train_data_loader, test_data_loader, is_train=True)

    # Build the model
    solver.build()

    # Train the model 
    solver.train()
    
    # Test the model
    #solver.eval(mode='test', to_print = False)
