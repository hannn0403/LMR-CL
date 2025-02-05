# solvers/solver_wav.py

import os
import math
import re
import pickle
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F

# 고정 시드 (재현성을 위함)
torch.manual_seed(123)
torch.cuda.manual_seed_all(123)

from utils.utils import to_gpu, time_desc_decorator
from models.wav import LMR  # models/wav.py에 정의된 LMR 모델을 사용

class Solver(object):
    def __init__(self, train_config, test_config, train_data_loader, test_data_loader, is_train=True, model=None):
        """
        Parameters:
          - train_config: 학습 설정(config) 객체 (배치 크기, learning_rate, optimizer 등 포함)
          - test_config: 평가용 config 객체
          - train_data_loader: 학습 데이터 로더
          - test_data_loader: 평가 데이터 로더
          - is_train: 학습 모드(True이면 학습 진행, False이면 평가만 수행)
          - model: 이미 생성된 모델 (없으면 build()에서 새로 생성)
        """
        self.train_config = train_config
        self.test_config  = test_config
        self.train_data_loader = train_data_loader
        self.test_data_loader  = test_data_loader
        self.is_train = is_train
        self.model = model

    @time_desc_decorator('Build Graph')
    def build(self, cuda=True):
        if self.model is None:
            self.model = LMR(self.train_config)
        # 모델을 지정된 device로 이동
        if torch.cuda.is_available() and cuda:
            self.model.to(self.train_config.device)
        # 학습 모드라면 optimizer 설정
        if self.is_train:
            self.optimizer = self.train_config.optimizer(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.train_config.learning_rate
            )
        # 예시: wav2vec freezing (필요에 따라 수정)
        for name, param in self.model.named_parameters():
            if "wav2vec2.feature" in name:
                param.requires_grad = False
            elif "wav2vec2.encoder.layers" in name:
                try:
                    layer_num = int(name.split("encoder.layers.")[1].split(".")[0])
                    if layer_num <= 21:
                        param.requires_grad = False
                except Exception as e:
                    pass

    @time_desc_decorator('Training Start!')
    def train(self):
        n_epoch = self.train_config.n_epoch
        curr_patience = self.train_config.patience
        num_trials = 1

        # 예시 loss weight (필요에 따라 조정)
        loss_weight = [0.97600584, 0.98948395, 0.99224594, 0.81871365, 0.01936299, 0.9798473, 0.97490966]
        loss_weight = torch.FloatTensor(loss_weight).to(self.train_config.device)
        self.criterion = nn.CrossEntropyLoss(loss_weight, reduction="mean")

        best_valid_f1 = 0.0
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.5)

        for e in range(n_epoch):
            self.model.train()
            train_loss_list = []
            y_pred_list = []
            y_true_list = []
            for batch in self.train_data_loader:
                self.model.zero_grad()
                # WAV 데이터의 배치는 (wav_input, wav_att_mask, labels) 형태로 구성됨.
                wav_input, wav_att_mask, y = batch
                wav_input = to_gpu(wav_input, gpu_id=0)
                wav_att_mask = to_gpu(wav_att_mask, gpu_id=0)
                y = to_gpu(y, gpu_id=0).squeeze()
                
                output = self.model(wav_input, wav_att_mask)
                # output가 ModelOutput 객체인 경우 logits 추출
                logits = output.logits if hasattr(output, 'logits') else output
                loss = self.criterion(logits, y)
                loss.backward()
                max_norm = 5.0
                torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, self.model.parameters()), max_norm)
                self.optimizer.step()
                
                train_loss_list.append(loss.item())
                y_pred_list.append(logits.detach().cpu().numpy())
                y_true_list.append(y.detach().cpu().numpy())
            
            y_true_concat = np.concatenate(y_true_list, axis=0).squeeze()
            y_pred_concat = np.concatenate(y_pred_list, axis=0).squeeze()
            print("Train Loss:", np.mean(train_loss_list))
            print("Training accuracy:", self.calc_metrics(y_true_concat, y_pred_concat, mode='train', to_print=False))
            
            valid_loss, valid_acc, valid_f1 = self.eval(mode="test", to_print=False)
            print(f"Current patience: {curr_patience}")
            if valid_f1 >= best_valid_f1:
                best_valid_f1 = valid_f1
                print("Found new best model on test set! F1:", best_valid_f1)
                if not os.path.exists('checkpoints'):
                    os.makedirs('checkpoints')
                torch.save(self.model.state_dict(), f'checkpoints/model_wav_{self.train_config.name}.std')
                torch.save(self.optimizer.state_dict(), f'checkpoints/optim_wav_{self.train_config.name}.std')
                curr_patience = self.train_config.patience
            else:
                curr_patience -= 1
                if curr_patience <= -1:
                    print("Patience expired, loading best model.")
                    num_trials -= 1
                    curr_patience = self.train_config.patience
                    self.model.load_state_dict(torch.load(f'checkpoints/model_wav_{self.train_config.name}.std'))
                    self.optimizer.load_state_dict(torch.load(f'checkpoints/optim_wav_{self.train_config.name}.std'))
                    lr_scheduler.step()
                    print("Current learning rate:", self.optimizer.state_dict()['param_groups'][0]['lr'])
            if num_trials <= 0:
                print("Early stopping.")
                break
        
        self.eval(mode="test", to_print=True)

    def eval(self, mode=None, to_print=False):
        assert mode is not None, "Mode must be specified ('train' or 'test')."
        self.model.eval()
        y_true_list = []
        y_pred_list = []
        eval_loss_list = []
        dataloader = self.test_data_loader if mode == "test" else self.train_data_loader
        if to_print:
            self.model.load_state_dict(torch.load(f'checkpoints/model_wav_{self.train_config.name}.std'))
        with torch.no_grad():
            for batch in dataloader:
                self.model.zero_grad()
                wav_input, wav_att_mask, y = batch
                wav_input = to_gpu(wav_input, gpu_id=0)
                wav_att_mask = to_gpu(wav_att_mask, gpu_id=0)
                y = to_gpu(y, gpu_id=0).squeeze()
                output = self.model(wav_input, wav_att_mask)
                logits = output.logits if hasattr(output, 'logits') else output
                loss = self.criterion(logits, y)
                eval_loss_list.append(loss.item())
                y_pred_list.append(logits.detach().cpu().numpy())
                y_true_list.append(y.detach().cpu().numpy())
        eval_loss = np.mean(eval_loss_list)
        y_true_concat = np.concatenate(y_true_list, axis=0).squeeze()
        y_pred_concat = np.concatenate(y_pred_list, axis=0).squeeze()
        print("######Test Confusion Matrix######")
        accuracy = self.calc_metrics(y_true_concat, y_pred_concat, mode, to_print=True)
        y_pred_arg = np.argmax(y_pred_concat, axis=1)
        f1 = f1_score(y_true_concat, y_pred_arg, average='macro')
        return eval_loss, accuracy, f1

    def calc_metrics(self, y_true, y_pred, mode=None, to_print=False):
        test_preds = np.argmax(y_pred, axis=1)
        test_truth = y_true
        if to_print:
            print("Confusion Matrix:")
            print(confusion_matrix(test_truth, test_preds))
            print("Classification Report:")
            print(classification_report(test_truth, test_preds, digits=5))
            print("Accuracy:", accuracy_score(test_truth, test_preds))
        return accuracy_score(test_truth, test_preds)
