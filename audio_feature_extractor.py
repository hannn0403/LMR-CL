import os
import pickle
import numpy as np
from random import random

from config import get_config, activation_dict
from data_loader_wav_20 import get_loader
from data_loader_wav_fintuning import get_loader_fine
from solver_wav_feature import Solver
from solver_wav_feature_fintuning import Solver_fine

import torch
import torch.nn as nn 
from torch.nn import functional as F



#torch.multiprocessing.set_start_method('spawn')
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "0,1"
#os.environ["CUDA_VISIBLE_DEVICES"] = "1" 

if __name__ == '__main__':
    #torch.autograd.set_detect_anomaly(True)
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
    train_data_loader = get_loader(train_config, shuffle = False)
    test_data_loader = get_loader(test_config, shuffle = False)
    
    train_data_loader_fine = get_loader_fine(train_config, shuffle = False)
    test_data_loader_fine = get_loader_fine(test_config, shuffle = False)


    # Solver is a wrapper for model traiing and testing
    
    solver_fine = Solver_fine
    solver_fine = solver_fine(train_config, test_config, train_data_loader_fine, test_data_loader_fine, is_train=True)
    

    # Build the model
   
    solver_fine.build()

    # Train the model
    solver_fine.train()
    
    torch.cuda.empty_cache()
    
    solver = Solver
    solver = solver(train_config, test_config, train_data_loader, test_data_loader, is_train=False)
    
    solver.eval(mode='train', to_print = False)
    solver.eval(mode='test', to_print = False)
  