# solvers/solver_bert.py

import os
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import to_gpu, time_desc_decorator, DiffLoss, CMD, IAMC, IEMC
from models.bert import LMR

# 시드 고정 (재현성을 위해)
torch.manual_seed(123)
torch.cuda.manual_seed_all(123)

class Solver(object):
    def __init__(self, train_config, test_config, train_data_loader, test_data_loader, is_train=True, model=None):
        """
        Parameters:
          - train_config: 학습 설정 객체 (config)
          - test_config: 평가 설정 객체 (config)
          - train_data_loader: 학습 데이터 로더
          - test_data_loader: 평가 데이터 로더
          - is_train: 학습 모드 여부 (True이면 학습, False이면 eval만 수행)
          - model: 사전 생성된 모델 (없으면 build()에서 새로 생성)
        """
        self.train_config = train_config
        self.test_config = test_config
        self.train_data_loader = train_data_loader
        self.test_data_loader = test_data_loader
        self.is_train = is_train
        self.model = model
        self.epoch_i = 0

    @time_desc_decorator('Build Graph')
    def build(self, cuda=True):
        if self.model is None:
            self.model = LMR(self.train_config)
        # 예시: BERT의 하위 레이어 동결 (필요에 따라 수정)
        for name, param in self.model.named_parameters():
            if "bertmodel.electra.encoder.layer" in name:
                try:
                    layer_num = int(name.split("encoder.layer.")[1].split(".")[0])
                    if layer_num <= 7:
                        param.requires_grad = False
                except Exception as e:
                    pass
            elif "bertmodel.electra.embeddings" in name:
                param.requires_grad = False
            if 'weight_hh' in name:
                nn.init.orthogonal_(param)
        if torch.cuda.is_available() and cuda:
            self.model.to(self.train_config.device)
        if self.is_train:
            self.optimizer = self.train_config.optimizer(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.train_config.learning_rate
            )

    @time_desc_decorator('Training Start!')
    def train(self):
        curr_patience = self.train_config.patience
        num_trials = 1
        # 예시 loss weight (상황에 맞게 수정)
        loss_weight = [0.97600584, 0.98948395, 0.99224594, 0.81871365, 0.01936299, 0.9798473, 0.97490966]
        loss_weight = torch.FloatTensor(loss_weight).to(self.train_config.device)
        self.criterion = nn.CrossEntropyLoss(loss_weight, reduction="mean")
        self.loss_diff = DiffLoss()
        self.loss_cmd = CMD()
        self.loss_acou_aux = nn.CrossEntropyLoss(loss_weight, reduction="mean")
        self.loss_text_aux = nn.CrossEntropyLoss(loss_weight, reduction="mean")
        self.loss_phy_aux = nn.CrossEntropyLoss(loss_weight, reduction="mean")
        self.loss_iamc = IAMC(256)
        self.loss_iemc = IEMC(256)
        
        best_valid_f1 = 0.0
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.5)
        
        for e in range(self.train_config.n_epoch):
            self.model.train()
            train_loss_cls = []
            train_loss_diff = []
            train_loss_sim = []
            train_loss_total = []
            y_pred_list = []
            y_true_list = []
            
            for batch in self.train_data_loader:
                self.model.zero_grad()
                # 배치 구성: (texts, phy, acoustic, labels, lengths, bert_sentences, bert_att_mask)
                t, p, w, y, l, bert_sent, bert_sent_mask = batch
                
                p = to_gpu(p, gpu_id=0)
                w = to_gpu(w, gpu_id=0)
                y = to_gpu(y, gpu_id=0).squeeze()
                bert_sent = to_gpu(bert_sent, gpu_id=0)
                bert_sent_mask = to_gpu(bert_sent_mask, gpu_id=0)
                
                y_tilde = self.model(t, p, w, l, bert_sent, bert_sent_mask)
                cls_loss = self.criterion(y_tilde, y)
                diff_loss = self.get_diff_loss()
                cmd_loss = self.get_cmd_loss()
                aux_phy_loss = self.get_auxiliary_phy_loss(y)
                aux_acou_loss = self.get_auxiliary_audio_loss(y)
                aux_text_loss = self.get_auxiliary_text_loss(y)
                iamc_loss = self.get_iamcl_loss(y)
                iemc_loss = self.get_iemcl_loss(y)
                
                loss = (cls_loss +
                        self.train_config.diff_weight * diff_loss +
                        self.train_config.sim_weight * cmd_loss +
                        0.1 * iamc_loss +
                        0.1 * iemc_loss +
                        aux_acou_loss + aux_text_loss + aux_phy_loss)
                
                loss.backward()
                max_norm = 5.0
                torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, self.model.parameters()), max_norm)
                self.optimizer.step()
                
                train_loss_cls.append(cls_loss.item())
                train_loss_diff.append(diff_loss.item())
                train_loss_sim.append(cmd_loss.item())
                train_loss_total.append(loss.item())
                
                y_pred_list.append(y_tilde.detach().cpu().numpy())
                y_true_list.append(y.detach().cpu().numpy())
            
            y_true_concat = np.concatenate(y_true_list, axis=0).squeeze()
            y_pred_concat = np.concatenate(y_pred_list, axis=0).squeeze()
            
            print('Train Loss (cls):', np.mean(train_loss_cls))
            print('Train Loss (diff):', np.mean(train_loss_diff))
            print('Train Loss (sim):', np.mean(train_loss_sim))
            print("Training accuracy:", self.calc_metrics(y_true_concat, y_pred_concat, mode='train', to_print=False))
            
            valid_loss, valid_acc, valid_f1 = self.eval(mode="test", to_print=False)
            print(f"Current patience: {curr_patience}")
            if valid_f1 >= best_valid_f1:
                best_valid_f1 = valid_f1
                print("Found new best model on test set! F1:", best_valid_f1)
                if not os.path.exists('checkpoints'):
                    os.makedirs('checkpoints')
                torch.save(self.model.state_dict(), f'checkpoints/model_bert_{self.train_config.name}.std')
                torch.save(self.optimizer.state_dict(), f'checkpoints/optim_bert_{self.train_config.name}.std')
                curr_patience = self.train_config.patience
            else:
                curr_patience -= 1
                if curr_patience <= -1:
                    print("Patience expired, loading best model.")
                    num_trials -= 1
                    curr_patience = self.train_config.patience
                    self.model.load_state_dict(torch.load(f'checkpoints/model_bert_{self.train_config.name}.std'))
                    self.optimizer.load_state_dict(torch.load(f'checkpoints/optim_bert_{self.train_config.name}.std'))
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
        eval_loss = []
        
        dataloader = self.test_data_loader if mode == "test" else self.train_data_loader
        
        if to_print:
            self.model.load_state_dict(torch.load(f'checkpoints/model_bert_{self.train_config.name}.std'))
        
        with torch.no_grad():
            for batch in dataloader:
                self.model.zero_grad()
                t, p, w, y, l, bert_sent, bert_sent_mask = batch
                p = to_gpu(p, gpu_id=0)
                w = to_gpu(w, gpu_id=0)
                y = to_gpu(y, gpu_id=0).squeeze()
                bert_sent = to_gpu(bert_sent, gpu_id=0)
                bert_sent_mask = to_gpu(bert_sent_mask, gpu_id=0)
                
                y_tilde = self.model(t, p, w, l, bert_sent, bert_sent_mask)
                cls_loss = self.criterion(y_tilde, y)
                eval_loss.append(cls_loss.item())
                y_pred_list.append(y_tilde.detach().cpu().numpy())
                y_true_list.append(y.detach().cpu().numpy())
        
        eval_loss = np.mean(eval_loss)
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

    def get_cmd_loss(self):
        if not self.train_config.use_cmd_sim:
            return 0.0
        loss1 = self.loss_cmd(self.model.utt_shared_t, self.model.utt_shared_p, 3)
        loss2 = self.loss_cmd(self.model.utt_shared_t, self.model.utt_shared_a, 3)
        loss3 = self.loss_cmd(self.model.utt_shared_a, self.model.utt_shared_p, 3)
        loss = (loss1 + loss2 + loss3) / 3.0
        return loss

    def get_diff_loss(self):
        shared_t = self.model.utt_shared_t
        shared_p = self.model.utt_shared_p
        shared_a = self.model.utt_shared_a
        private_t = self.model.utt_private_t
        private_p = self.model.utt_private_p
        private_a = self.model.utt_private_a
        loss = self.loss_diff(private_t, shared_t)
        loss += self.loss_diff(private_p, shared_p)
        loss += self.loss_diff(private_a, shared_a)
        loss += self.loss_diff(private_a, private_t)
        loss += self.loss_diff(private_a, private_p)
        loss += self.loss_diff(private_t, private_p)
        return loss

    def get_auxiliary_phy_loss(self, y):
        phy_predict = self.model.phy_output
        loss = self.loss_phy_aux(phy_predict, y)
        return loss

    def get_auxiliary_audio_loss(self, y):
        acou_predict = self.model.acou_output
        loss = self.loss_acou_aux(acou_predict, y)
        return loss

    def get_auxiliary_text_loss(self, y):
        text_predict = self.model.text_output
        loss = self.loss_text_aux(text_predict, y)
        return loss

    def get_iamcl_loss(self, y):
        loss = self.loss_iamc(
            y,
            self.model.utt_private_t, self.model.utt_private_p, self.model.utt_private_a,
            self.model.utt_shared_t, self.model.utt_shared_p, self.model.utt_shared_a
        )
        return loss

    def get_iemcl_loss(self, y):
        loss = self.loss_iemc(y, self.model.utt_shared_t, self.model.utt_shared_p, self.model.utt_shared_a)
        return loss
