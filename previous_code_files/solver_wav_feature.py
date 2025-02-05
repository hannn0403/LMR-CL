import os
import math
from math import isnan
import re
import pickle
import gensim
import numpy as np
from tqdm import tqdm
from tqdm import tqdm_notebook
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from scipy.special import expit

import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(123)
torch.cuda.manual_seed_all(123)

from utils import to_gpu, time_desc_decorator
import models_only_wav_fintuning

import matplotlib.pyplot as plt
from transformers import Wav2Vec2Model
import pandas as pd


class Solver(object):
    def __init__(self, train_config, test_config, train_data_loader, test_data_loader, is_train=True, model=None):

        self.train_config = train_config
        self.epoch_i = 0
        self.train_data_loader = train_data_loader
        self.test_data_loader = test_data_loader
        self.is_train = is_train

    
    @time_desc_decorator('Build Graph')
    def build(self, cuda=True):

        if self.model is None:
            self.model = getattr(models_only_wav_fintuning, self.train_config.model)(self.train_config)
            
        if torch.cuda.is_available() and cuda:
            self.model.cuda()

        
        # Final list
        for name, param in self.model.named_parameters():

            # wav2vec freezing customizations 
            if self.train_config.data == "kemdy20":
   
                if "wav2model.feature" in name:
                    param.requires_grad = False
                    
                elif "wav2model.encoder.layers" in name:
                    layer_num = int(name.split("encoder.layers.")[-1].split(".")[0])
                    if layer_num <= (22):
                        param.requires_grad = False
                elif "wav2model.encoder.pos_conv_embed" in name:
                    param.requires_grad = False
                    
                elif "wav2model.encoder.layer_norm" in name:
                    param.requires_grad = False
                        
            
            if 'weight_hh' in name:
                nn.init.orthogonal_(param)
            print('\t' + name, param.requires_grad) 

        
        if self.is_train:
            self.optimizer = self.train_config.optimizer(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.train_config.learning_rate)



    def eval(self,mode=None, to_print=False):
        assert(mode is not None)
        pre_wav = Wav2Vec2Model.from_pretrained("kresnik/wav2vec2-large-xlsr-korean").cuda()
        pre_wav.load_state_dict(torch.load('checkpoints/model_wav_fine.std')) 
        
        pre_wav.eval()

   
        if mode == "test":
            dataloader = self.test_data_loader

        elif mode == "train":
            dataloader = self.train_data_loader

            
        wav_feature_list = []
        with torch.no_grad(): 
            for batch in dataloader:
                pre_wav.zero_grad()
                wav2_input, wav2_mask, y = batch

                wav2_input = to_gpu(wav2_input, gpu_id=0)
                wav2_mask = to_gpu(wav2_mask, gpu_id=0)
     
                feature_outputs = pre_wav(wav2_input, wav2_mask)
                last_hidden_states = feature_outputs.last_hidden_state

                last_hidden_states = torch.sum(last_hidden_states, dim=1, keepdim=False) / last_hidden_states.shape[1]
                wav_feature_list.append(last_hidden_states.detach().cpu().numpy())
    
 
        np.save('./Data/wav_feature_20_fintuning_{}.npy'.format(mode), np.array(wav_feature_list))    
        

    

        

   





