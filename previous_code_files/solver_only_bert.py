import os
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
from torch.nn import functional as F

from utils import to_gpu, time_desc_decorator, DiffLoss, CMD, IAMC, IEMC
import models_only_bert

torch.manual_seed(123)
torch.cuda.manual_seed_all(123)


class Solver(object):
    def __init__(self, train_config, test_config, train_data_loader, test_data_loader, is_train=True, model=None):

        self.train_config = train_config
        self.epoch_i = 0
        self.train_data_loader = train_data_loader
        self.test_data_loader = test_data_loader
        self.is_train = is_train
        self.model = model
    
    @time_desc_decorator('Build Graph') 
    def build(self, cuda=True):
        #os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
        #os.environ["CUDA_VISIBLE_DEVICES"]= "1"

        if self.model is None:
            self.model = getattr(models_only_bert, self.train_config.model)(self.train_config)
        
        # Final list
        for name, param in self.model.named_parameters():

            # electra freezing customizations        
            if "bertmodel.electra.encoder.layer" in name:
                layer_num = int(name.split("encoder.layer.")[-1].split(".")[0])
                if layer_num <= (7):
                    param.requires_grad = False
                        
            elif "bertmodel.electra.embeddings" in name:
                param.requires_grad = False
                
            if 'weight_hh' in name:
                nn.init.orthogonal_(param)
            #print('\t' + name, param.requires_grad)

        
        if torch.cuda.is_available() and cuda:
            self.model.to('cuda:0')

        if self.is_train:
            self.optimizer = self.train_config.optimizer(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.train_config.learning_rate)


    @time_desc_decorator('Training Start!')
    def train(self):
        curr_patience = patience = self.train_config.patience
        num_trials = 1

        loss_weight = [0.97600584, 0.98948395, 0.99224594, 0.81871365, 0.01936299, 0.9798473 , 0.97490966]

        loss_weight = torch.FloatTensor(loss_weight).to('cuda:0')
        self.criterion = criterion = nn.CrossEntropyLoss(loss_weight, reduction="mean")
        self.loss_diff = DiffLoss()
        self.loss_cmd = CMD()
        
        self.loss_acou_aux = nn.CrossEntropyLoss(loss_weight, reduction="mean")
        self.loss_text_aux = nn.CrossEntropyLoss(loss_weight, reduction="mean")
        self.loss_phy_aux = nn.CrossEntropyLoss(loss_weight, reduction="mean")

        self.loss_iamc = IAMC(256)
        self.loss_iemc = IEMC(256) 
        
        best_valid_f1 = 0.0
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.5)
        
        graph_diff_losses = []
        graph_sim_losses = []
        graph_cls_losses = []
     
        train_losses = []

        for e in range(self.train_config.n_epoch):
            self.model.train()

            train_loss_cls, train_loss_sim, train_loss_diff = [], [], []
            y_pred = []
            y_true = []

            train_loss = []

            for batch in self.train_data_loader:
                self.model.zero_grad()
                t, p, w, y, l, bert_sent, bert_sent_mask = batch

                p = to_gpu(p, gpu_id=0)
                w = to_gpu(w, gpu_id=0)
                y = to_gpu(y, gpu_id=0)
                y = y.squeeze()
    
                bert_sent = to_gpu(bert_sent, gpu_id=0)
                bert_sent_mask = to_gpu(bert_sent_mask, gpu_id=0)

                y_tilde = self.model(t, p, w, l, bert_sent, bert_sent_mask)
                print_tf = False

                cls_loss = criterion(y_tilde, y)
                diff_loss = self.get_diff_loss()

                cmd_loss = self.get_cmd_loss(print_tf)
                aux_phy_loss = self.get_auxiliary_phy_loss(y) 
                aux_acou_loss = self.get_auxiliary_audio_loss(y)
                aux_text_loss = self.get_auxiliary_text_loss(y)
                
                iamcl_loss = self.get_iamcl_loss(y)
                iemcl_loss = self.get_iemcl_loss(y)
                
                if self.train_config.use_cmd_sim:
                    similarity_loss = cmd_loss
                else:
                    similarity_loss = None
                
                loss = cls_loss + self.train_config.diff_weight * diff_loss + self.train_config.sim_weight * similarity_loss + 0.1*iemcl_loss + 0.1*iamcl_loss + aux_acou_loss + aux_text_loss + aux_phy_loss

                loss.backward()
                
                max_norm = 5.0
                torch.nn.utils.clip_grad_norm_([param for param in self.model.parameters() if param.requires_grad], max_norm)

                self.optimizer.step()

                train_loss_cls.append(cls_loss.item())
                train_loss_diff.append(diff_loss.item())
                train_loss.append(loss.item())
                train_loss_sim.append(similarity_loss.item())

                y_pred.append(y_tilde.detach().cpu().numpy())
                y_true.append(y.detach().cpu().numpy())
                
                
            y_true = np.concatenate(y_true, axis=0).squeeze()
            y_pred = np.concatenate(y_pred, axis=0).squeeze()
            
            m_tlc = np.mean(train_loss_cls, axis=0)
            m_tls = np.mean(train_loss_sim, axis=0)
            m_tld = np.mean(train_loss_diff, axis=0)
    
            print('train_loss_cls', m_tlc)
            print('train_loss_sim', m_tls)
            print('train_loss_diff', m_tld)
       
            graph_cls_losses.append(m_tlc)
            graph_sim_losses.append(m_tls)
            graph_diff_losses.append(m_tld)

         
            accuracy = self.calc_metrics(y_true, y_pred, mode='train', to_print = False)
            print("Training accuracy: ", accuracy)
           

            train_losses.append(train_loss)
            print(f"Training loss: {round(np.mean(train_loss), 4)}")

            valid_loss, valid_acc, valid_f1 = self.eval(mode="test", to_print=False)
            
            print(f"Current patience: {curr_patience}, current trial: {num_trials}.")
            if valid_f1 >= best_valid_f1:
                best_valid_f1 = valid_f1
                print("Found new best model on test set! f1 score: ", best_valid_f1)

                if not os.path.exists('checkpoints'): os.makedirs('checkpoints')
                torch.save(self.model.state_dict(), f'checkpoints/model_sim_diff_gating_iam_iem_{self.train_config.name}.std')
                torch.save(self.optimizer.state_dict(), f'checkpoints/optim_sim_diff_gating_iam_iem_{self.train_config.name}.std')
                curr_patience = patience
            else:
                curr_patience -= 1
                if curr_patience <= -1:
                    print("Running out of patience, loading previous best model.")
                    num_trials -= 1
                    curr_patience = patience
                    self.model.load_state_dict(torch.load(f'checkpoints/model_sim_diff_gating_iam_iem_{self.train_config.name}.std'))
                    self.optimizer.load_state_dict(torch.load(f'checkpoints/optim_sim_diff_gating_iam_iem_{self.train_config.name}.std'))
                    lr_scheduler.step()
                    print(f"Current learning rate: {self.optimizer.state_dict()['param_groups'][0]['lr']}")
            
            if num_trials <= 0:
                print("Running out of patience, early stopping.")
                break

        self.eval(mode="test", to_print=True)

    
    def eval(self,mode=None, to_print=False):
        assert(mode is not None)
        self.model.eval()

        y_true, y_pred = [], []
        eval_loss = []


        if mode == "test":
            dataloader = self.test_data_loader

            if to_print:
                self.model.load_state_dict(torch.load(
                    f'checkpoints/model_sim_diff_gating_iam_iem_{self.train_config.name}.std'))
            

        with torch.no_grad():

            for batch in dataloader:
                self.model.zero_grad()
                t, p, w, y, l, bert_sent, bert_sent_mask = batch

                p = to_gpu(p, gpu_id=0)
                w = to_gpu(w, gpu_id=0)
                y = to_gpu(y, gpu_id=0)
                y = y.squeeze()

                bert_sent = to_gpu(bert_sent, gpu_id=0)
                bert_sent_mask = to_gpu(bert_sent_mask, gpu_id=0)

                y_tilde = self.model(t, p, w, l, bert_sent, bert_sent_mask)
               
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

    def multiclass_acc(self, preds, truths):
        """
        Compute the multiclass accuracy w.r.t. groundtruth
        :param preds: Float array representing the predictions, dimension (N,)
        :param truths: Float/int array representing the groundtruth classes, dimension (N,)
        :return: Classification accuracy
        """
        return np.sum(np.round(preds) == np.round(truths)) / float(len(truths))

    def calc_metrics(self, y_true, y_pred, mode=None, to_print=False):
        """
        Metric scheme adapted from:
        https://github.com/yaohungt/Multimodal-Transformer/blob/master/src/eval_metrics.py
        """

       
        test_preds = np.argmax(y_pred, 1)
        test_truth = y_true

        if to_print:
            print("Confusion Matrix (pos/neg) :")
            print(confusion_matrix(test_truth, test_preds))
            print("Classification Report (pos/neg) :")
            print(classification_report(test_truth, test_preds, digits=5))
            print("Accuracy (pos/neg) ", accuracy_score(test_truth, test_preds))
            
        return accuracy_score(test_truth, test_preds)


    def get_cmd_loss(self,print_tf=False):

        if not self.train_config.use_cmd_sim:
            return 0.0

        # losses between shared states
        loss1 = self.loss_cmd(self.model.utt_shared_t, self.model.utt_shared_p, 3)
        loss2 = self.loss_cmd(self.model.utt_shared_t, self.model.utt_shared_a, 3)
        loss3 = self.loss_cmd(self.model.utt_shared_a, self.model.utt_shared_p, 3)
  
        loss = (loss1+loss2+loss3)/3.0

        return loss

    def get_diff_loss(self):

        shared_t = self.model.utt_shared_t
        shared_p = self.model.utt_shared_p
        shared_a = self.model.utt_shared_a
        private_t = self.model.utt_private_t
        private_p = self.model.utt_private_p
        private_a = self.model.utt_private_a

        # Between private and shared
        loss = self.loss_diff(private_t, shared_t)
        loss += self.loss_diff(private_p, shared_p)
        loss += self.loss_diff(private_a, shared_a)

        # Across privates
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

        loss = self.loss_iamc(y, self.model.utt_private_t, self.model.utt_private_p, self.model.utt_private_a, self.model.utt_shared_t, self.model.utt_shared_p, self.model.utt_shared_a)

        return loss


    def get_iemcl_loss(self, y):

        loss = self.loss_iemc(y, self.model.utt_shared_t,self.model.utt_shared_p, self.model.utt_shared_a)

        return loss




    

    





