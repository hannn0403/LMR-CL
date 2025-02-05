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

from utils import to_gpu, time_desc_decorator, DiffLoss, MSE, SIMSE, CMD, F1_Loss
import models_only_wav_fintuning

import matplotlib.pyplot as plt

import pandas as pd


class Solver_fine(object):
    def __init__(self, train_config, test_config, train_data_loader, test_data_loader, is_train=True, model=None):

        self.train_config = train_config
        self.epoch_i = 0
        self.train_data_loader = train_data_loader
        self.test_data_loader = test_data_loader
        self.is_train = is_train
        self.model = model
        
    @time_desc_decorator('Build Graph')
    def build(self, cuda=True):

        if self.model is None:
            self.model = getattr(models_only_wav_fintuning, self.train_config.model)(self.train_config)
            
        if torch.cuda.is_available() and cuda:
            self.model.cuda()
            self.model = nn.DataParallel(self.model, device_ids = [0, 1]).to('cuda')
            self.model.to('cuda')
            #print('=============check misa', self.model.module)
        
        # Final list
        for name, param in self.model.named_parameters():

            # Wav2vec freezing customizations 
            if self.train_config.data == "kemdy20":

                if "wav2vec2.feature" in name:
                    param.requires_grad = False
                    
                elif "wav2vec2.encoder.layers" in name:
                    layer_num = int(name.split("encoder.layers.")[-1].split(".")[0])
                    if layer_num <= (21):
                        param.requires_grad = False
                elif "wav2vec2.encoder.pos_conv_embed" in name:
                    param.requires_grad = False
                    
                elif "wav2vec2.encoder.layer_norm" in name:
                    param.requires_grad = False
                        
            
            if 'weight_hh' in name:
                nn.init.orthogonal_(param)
            print('\t' + name, param.requires_grad)


        
        if self.is_train:
            self.optimizer = self.train_config.optimizer(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.train_config.learning_rate)

    @time_desc_decorator('Training Start!')
    def train(self):
    
        n_epoch = 2
        
        curr_patience = patience = self.train_config.patience
        num_trials = 1

 
        loss_weight = [0.97600584, 0.98948395, 0.99224594, 0.81871365, 0.01936299, 0.9798473 , 0.97490966]
        loss_weight = torch.FloatTensor(loss_weight).to('cuda:0')
        self.criterion = criterion = nn.CrossEntropyLoss(loss_weight, reduction="mean")
        
        best_valid_loss = float('inf')
        best_valid_f1 = 0.0
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.5)
        
        
    
        
        train_losses = []
        valid_losses = []
        for e in range(n_epoch):
            self.model.train()

            train_loss_cls, train_loss_sim, train_loss_diff = [], [], []
            y_pred = []
            y_true = []

            train_loss_sp = []
            train_loss = []
            train_loss_iam = []
            train_loss_iem = []
            for batch in self.train_data_loader:
                self.model.zero_grad()
                w_input, w_mask, y = batch
                
                w_input = to_gpu(w_input, gpu_id=0)
                w_mask = to_gpu(w_mask, gpu_id=0)
                y = to_gpu(y, gpu_id=0)
                y = y.squeeze()
               

                y_tilde = self.model(w_input, w_mask)
                print_tf = False
                y_tilde = y_tilde.logits


                cls_loss = criterion(y_tilde, y)
                
              
                
                loss = cls_loss 
                loss.backward()
                
                max_norm = 5.0
                torch.nn.utils.clip_grad_norm_([param for param in self.model.parameters() if param.requires_grad], max_norm)
      
                self.optimizer.step()

                train_loss_cls.append(cls_loss.item())
                train_loss.append(loss.item())
    

                y_pred.append(y_tilde.detach().cpu().numpy())
                y_true.append(y.detach().cpu().numpy())
                    
            y_true = np.concatenate(y_true, axis=0).squeeze()
            y_pred = np.concatenate(y_pred, axis=0).squeeze()
            
            m_tlc = np.mean(train_loss_cls, axis=0)
            
            
            print('train_loss_cls', m_tlc)

            

            #print('######Training Confusion Matrix######')
            accuracy = self.calc_metrics(y_true, y_pred, mode='train', to_print = False)
            print("Training accuracy: ", accuracy)
           

            train_losses.append(train_loss)
            print(f"Training loss: {round(np.mean(train_loss), 4)}")

            valid_loss, valid_acc, valid_f1 = self.eval(mode="test", to_print=False)
            
            print(f"Current patience: {curr_patience}, current trial: {num_trials}.")
            if valid_f1 >= best_valid_f1:
                best_valid_f1 = valid_f1
                print("Found new best model on dev set! f1 score: ", best_valid_f1)

                if not os.path.exists('checkpoints'): os.makedirs('checkpoints')
                #torch.save(self.model.state_dict(), f'checkpoints/model_wav_fine_{self.train_config.name}.std')
                #torch.save(self.model.wav_fintuing.wav2vec2.state_dict(), f'checkpoints/llast_model_wav_fine_{self.train_config.name}.std')
                #torch.save(self.model.state_dict(), f'checkpoints/model_wav_fine_{self.train_config.name}.std')
                #torch.save(self.optimizer.state_dict(), f'checkpoints/optim_wav_fine_{self.train_config.name}.std')
                
                #torch.save(self.model.wav_fintuing.wav2vec2.state_dict(), f'checkpoints/model_wav_fine.std')
                #torch.save(self.model.state_dict(), f'checkpoints/model_wav_fine.std')
                torch.save(self.model.module.wav_fintuing.wav2vec2.state_dict(), f'checkpoints/model_wav_fine.std')
                torch.save(self.optimizer.state_dict(), f'checkpoints/optim_wav_fine.std')
                curr_patience = patience
            else:
                curr_patience -= 1
                if curr_patience <= -1:
                    print("Running out of patience, loading previous best model.")
                    num_trials -= 1
                    curr_patience = patience
                    #self.model.load_state_dict(torch.load(f'checkpoints/model_wav_fine_{self.train_config.name}.std'))
                    #self.optimizer.load_state_dict(torch.load(f'checkpoints/optim_wav_fine_{self.train_config.name}.std'))
                    #self.model.load_state_dict(torch.load(f'checkpoints/model_wav_fine.std'))
                    #self.optimizer.load_state_dict(torch.load(f'checkpoints/optim_wav_fine.std'))
                    lr_scheduler.step()
                    print(f"Current learning rate: {self.optimizer.state_dict()['param_groups'][0]['lr']}")
            
            if num_trials <= 0:
                print("Running out of patience, early stopping.")
                break

        #self.eval(mode="test", to_print=False)

    def eval(self,mode=None, to_print=False):
        assert(mode is not None)
        self.model.eval()

        y_true, y_pred = [], []
        eval_loss, eval_loss_diff = [], []

        if mode == "test":
            dataloader = self.test_data_loader

            if to_print:
                self.model.load_state_dict(torch.load(
                    f'checkpoints/model_wav_fine.std'))
            
        
        with torch.no_grad():

            for batch in dataloader:
                self.model.zero_grad()
                w_input, w_mask, y = batch

                w_input = to_gpu(w_input, gpu_id=0)
                w_mask = to_gpu(w_mask, gpu_id=0)
                y = to_gpu(y, gpu_id=0)
                y = y.squeeze()
 

                y_tilde = self.model(w_input, w_mask)
                y_tilde = y_tilde.logits
    

                cls_loss = self.criterion(y_tilde, y)
                loss = cls_loss

                eval_loss.append(loss.item())
                y_pred.append(y_tilde.detach().cpu().numpy())
                y_true.append(y.detach().cpu().numpy())

        eval_loss = np.mean(eval_loss)
        y_true = np.concatenate(y_true, axis=0).squeeze()
        y_pred = np.concatenate(y_pred, axis=0).squeeze()

        print('######Test Confusion Matrix######')
        accuracy = self.calc_metrics(y_true, y_pred, mode, to_print = True)
        y_pred_arg = np.argmax(y_pred, 1)
        f1 = f1_score(y_true, y_pred_arg, average='macro')
        

        return eval_loss, accuracy, f1
    
 
         

    def calc_metrics(self, y_true, y_pred, mode=None, to_print=False):
        """
        Metric scheme adapted from:
        https://github.com/yaohungt/Multimodal-Transformer/blob/master/src/eval_metrics.py
        """


        if self.train_config.data == "kemdy20":
            test_preds = np.argmax(y_pred, 1)
            test_truth = y_true

            if to_print:
                print("Confusion Matrix (pos/neg) :")
                print(confusion_matrix(test_truth, test_preds))
                print("Classification Report (pos/neg) :")
                print(classification_report(test_truth, test_preds, digits=5))
                print("Accuracy (pos/neg) ", accuracy_score(test_truth, test_preds))
            
            return accuracy_score(test_truth, test_preds)

        else:
            test_preds = y_pred
            test_truth = y_true

            non_zeros = np.array([i for i, e in enumerate(test_truth) if e != 0])

            test_preds_a7 = np.clip(test_preds, a_min=-3., a_max=3.)
            test_truth_a7 = np.clip(test_truth, a_min=-3., a_max=3.)
            test_preds_a5 = np.clip(test_preds, a_min=-2., a_max=2.)
            test_truth_a5 = np.clip(test_truth, a_min=-2., a_max=2.)

            mae = np.mean(np.absolute(test_preds - test_truth))   # Average L1 distance between preds and truths
            corr = np.corrcoef(test_preds, test_truth)[0][1]
            mult_a7 = self.multiclass_acc(test_preds_a7, test_truth_a7)
            mult_a5 = self.multiclass_acc(test_preds_a5, test_truth_a5)
            
            f_score = f1_score((test_preds[non_zeros] > 0), (test_truth[non_zeros] > 0), average='weighted')
            
            # pos - neg
            binary_truth = (test_truth[non_zeros] > 0)
            binary_preds = (test_preds[non_zeros] > 0)

            if to_print:
                print("mae: ", mae)
                print("corr: ", corr)
                print("mult_acc: ", mult_a7)
                print("Classification Report (pos/neg) :")
                print(classification_report(binary_truth, binary_preds, digits=5))
                print("Accuracy (pos/neg) ", accuracy_score(binary_truth, binary_preds))
            
            # non-neg - neg
            binary_truth = (test_truth >= 0)
            binary_preds = (test_preds >= 0)

            if to_print:
                print("Classification Report (non-neg/neg) :")
                print(classification_report(binary_truth, binary_preds, digits=5))
                print("Accuracy (non-neg/neg) ", accuracy_score(binary_truth, binary_preds))
            
            return accuracy_score(binary_truth, binary_preds)
        

    

        

   





