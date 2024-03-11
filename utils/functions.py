from torch.autograd import Function
import torch.nn as nn
import torch
import random
import math
import numpy as np
from torch.nn import functional as F

"""
Adapted from https://github.com/fungtion/DSN/blob/master/functions.py
"""

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, p):
        ctx.p = p

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.p

        return output, None


class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, -pred)
        n = torch.numel(diffs.data)
        mse = torch.sum(diffs.pow(2)) / n

        return mse


class SIMSE(nn.Module):

    def __init__(self):
        super(SIMSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, - pred)
        n = torch.numel(diffs.data)
        simse = torch.sum(diffs).pow(2) / (n ** 2)

        return simse


class DiffLoss(nn.Module):

    def __init__(self):
        super(DiffLoss, self).__init__()

    def forward(self, input1, input2):

        batch_size = input1.size(0)
        input1 = input1.view(batch_size, -1)
        input2 = input2.view(batch_size, -1)

        # Zero mean
        input1_mean = torch.mean(input1, dim=0, keepdims=True)
        input2_mean = torch.mean(input2, dim=0, keepdims=True)
        input1 = input1 - input1_mean
        input2 = input2 - input2_mean

        input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True).detach()
        input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6)
        

        input2_l2_norm = torch.norm(input2, p=2, dim=1, keepdim=True).detach()
        input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-6)
        

        diff_loss = torch.mean((input1_l2.t().mm(input2_l2)).pow(2))

        return diff_loss

class CMD(nn.Module):
    """
    Adapted from https://github.com/wzell/cmd/blob/master/models/domain_regularizer.py
    """

    def __init__(self):
        super(CMD, self).__init__()

    def forward(self, x1, x2, n_moments):
        mx1 = torch.mean(x1, 0)
        mx2 = torch.mean(x2, 0)
        sx1 = x1-mx1
        sx2 = x2-mx2
 
        dm = torch.norm(mx1-mx2) + 1e-6
        scms = dm
        for i in range(n_moments - 1):
            scms += self.scm(sx1, sx2, i + 2)
        return scms

    def matchnorm(self, x1, x2):
        power = torch.pow((x1-x2), 2)
        summed = torch.sum(power)
        sqrt = summed**(0.5)
        return sqrt
      

    def scm(self, sx1, sx2, k):
        ss1 = torch.mean(torch.pow(sx1, k), 0)
        ss2 = torch.mean(torch.pow(sx2, k), 0)
        return torch.norm(ss1-ss2) + 1e-6
     



    

class F1_Loss(nn.Module):
    '''Calculate F1 score. Can work with gpu tensors
    The original implmentation is written by Michal Haltuf on Kaggle.
    Returns
    -------
    torch.Tensor
        `ndim` == 1. epsilon <= val <= 1
    Reference
    ---------
    - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6
    - http://www.ryanzhang.info/python/writing-your-own-loss-function-module-for-pytorch/
    '''
    def __init__(self, epsilon=1e-7):
        super().__init__()
        self.epsilon = epsilon
        
    def forward(self, y_pred, y_true,):
        assert y_pred.ndim == 2
        assert y_true.ndim == 1
        y_true = F.one_hot(y_true, 7).to(torch.float32)
        y_pred = F.softmax(y_pred, dim=1)
        tp = (y_true * y_pred).sum(dim=0).to(torch.float32)
        tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0).to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float32)
        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)
        f1 = 2* (precision*recall) / (precision + recall + self.epsilon)
        f1 = f1.clamp(min=self.epsilon, max=1-self.epsilon)
        return 1 - f1.mean()

class IAMC_nonphy(nn.Module):

    def __init__(self, batch_size):
        super(IAMC_nonphy, self).__init__()
        self.batch_size = batch_size 
        # 처음에는 0으로 설정 
        # a : label 6개 중에서 어떤 것을 할 것인지에 대한 인덱스
        self.a = 0
        self.label_list = ['angry','disgust','fear','happy','neutral','sad','surprise']

    def forward(self, label, p_t, p_a, s_t, s_a):

        # label : batch에서 y들에 대한 list 
        # a_m : 기준이 되는 modality
        # m_1, m_2 : 대상이 되는 modality 1, 2
        # anchor_label : a의 label을 가지고 있어서 anchor가 된 sample의 인덱스
        # anchor 
        # label에서  label_index와 동일한 것만을 가진 index 추출 
        indices = torch.nonzero(label == self.a).squeeze(1)
        # anchor의 label이 batch 내에서 유일한 경우에는 다시 뽑는다.
         
        while (indices.size(0)<=1):
            self.a = (self.a + 1) % 7
            # label에서  label_index와 동일한 것만을 가진 index 추출 
            indices = torch.nonzero(label == self.a).squeeze(1)

        # indices가 0이 아니라면 while문을 빠져 나옴
        # 추출한 것에서 랜덤하게 하나 뽑아서 anchor_label로 설정
        random_index = torch.randint(0, indices.size(0), (1,))
        anchor_index = indices[random_index.item()]
        
        
        anchor_label = self.label_list[self.a]

        loss_permodality = []
        
        re_finement_term_pt = []
        re_finement_term_pa = []
        re_finement_term_ct = []
        re_finement_term_ca = []

        re_finement_term_list = []


        pos_pt_score = []
        pos_pa_score = []
        pos_ct_score = []
        pos_ca_score = []
        
        neg_pt_score = []
        neg_pa_score = []
        neg_ct_score = []
        neg_ca_score = []
        
        
                    
        anchor_pt = p_t[anchor_index]
        anchor_pa = p_a[anchor_index]
        anchor_ct = s_t[anchor_index]
        anchor_ca = s_a[anchor_index]
        
        
        # Positive sample
       
        
        for sample_idx in range(label.size(0)):

           
            
            if sample_idx == anchor_index:
                continue
            
            else: 
       
                if self.a == label[sample_idx]:
                    
                    pos_pt_score.append(torch.dot(anchor_pt, p_t[sample_idx])/(torch.norm(anchor_pt)*torch.norm(p_t[sample_idx])))
                    re_finement_term_pt.append(torch.pow((torch.dot(anchor_pt, p_t[sample_idx])-1),2))

                    pos_pa_score.append(torch.dot(anchor_pa, p_a[sample_idx])/(torch.norm(anchor_pa)*torch.norm(p_a[sample_idx])))
                    re_finement_term_pa.append(torch.pow((torch.dot(anchor_pt, p_a[sample_idx])-1),2))
                    
                    pos_ct_score.append(torch.dot(anchor_ct, s_t[sample_idx])/(torch.norm(anchor_ct)*torch.norm(s_t[sample_idx])))
                    re_finement_term_ct.append(torch.pow((torch.dot(anchor_pt, s_t[sample_idx])-1),2))
                    
                    pos_ca_score.append(torch.dot(anchor_ca, s_a[sample_idx])/(torch.norm(anchor_ca)*torch.norm(s_a[sample_idx])))
                    re_finement_term_ca.append(torch.pow((torch.dot(anchor_pt, s_a[sample_idx])-1),2))
                    
                    
                else:
                    
                    neg_pt_score.append((torch.dot(anchor_pt, p_t[sample_idx])/(torch.norm(anchor_pt)*torch.norm(p_t[sample_idx]))))
                    neg_pa_score.append((torch.dot(anchor_pa, p_a[sample_idx])/(torch.norm(anchor_pa)*torch.norm(p_a[sample_idx]))))
                    
                    neg_ct_score.append((torch.dot(anchor_ct, s_t[sample_idx])/(torch.norm(anchor_ct)*torch.norm(s_t[sample_idx]))))
                    neg_ca_score.append((torch.dot(anchor_ca, s_a[sample_idx])/(torch.norm(anchor_ca)*torch.norm(s_a[sample_idx]))))

        
        pos_pt_score = 0 if len(pos_pt_score)==0 else (sum(pos_pt_score) / len(pos_pt_score))
        pos_pa_score = 0 if len(pos_pa_score)==0 else (sum(pos_pa_score) / len(pos_pa_score))
        pos_ct_score = 0 if len(pos_ct_score)==0 else (sum(pos_ct_score) / len(pos_ct_score))
        pos_ca_score = 0 if len(pos_ca_score)==0 else (sum(pos_ca_score) / len(pos_ca_score))
        neg_pt_score = 0 if len(neg_pt_score)==0 else (sum(neg_pt_score) / len(neg_pt_score))
        neg_pa_score = 0 if len(neg_pa_score)==0 else (sum(neg_pa_score) / len(neg_pa_score))
        neg_ct_score = 0 if len(neg_ct_score)==0 else (sum(neg_ct_score) / len(neg_ct_score))
        neg_ca_score = 0 if len(neg_ca_score)==0 else (sum(neg_ca_score) / len(neg_ca_score))

        re_finement_term_pt = 0 if len(re_finement_term_pt)==0 else (sum(re_finement_term_pt) / len(re_finement_term_pt))
        
        re_finement_term_pa = 0 if len(re_finement_term_pa)==0 else (sum(re_finement_term_pa) / len(re_finement_term_pa))
        re_finement_term_ct = 0 if len(re_finement_term_ct)==0 else (sum(re_finement_term_ct) / len(re_finement_term_ct))
    
        re_finement_term_ca = 0 if len(re_finement_term_ca)==0 else (sum(re_finement_term_ca) / len(re_finement_term_ca))
        
        re_finement_term_list.append(re_finement_term_pt)
        
        re_finement_term_list.append(re_finement_term_pa)
        re_finement_term_list.append(re_finement_term_ct)
        
        re_finement_term_list.append(re_finement_term_ca)

        

        loss_permodality.append(2-pos_pt_score+neg_pt_score) # private_t
        
        loss_permodality.append(2-pos_pa_score+neg_pa_score) # private_t


        loss_permodality.append(2-pos_ct_score+neg_ct_score) # common_t
        
        loss_permodality.append(2-pos_ca_score+neg_ca_score) # common_t

        
        
        iamcl_loss = sum(loss_permodality) / len(loss_permodality)
        re_finement_loss = sum(re_finement_term_list)/len(re_finement_term_list)

        # IEMCL과는 달리 IAMCL은 배치마다 1번만 돌아가므로 forward에서 self.a = (self.a + 1) % 7

        self.a = (self.a + 1) % 7

        return iamcl_loss  # -1*torch.as_tensor(iamcl_loss) #+ 0.3*re_finement_loss


class IEMC_nonphy(nn.Module):

    def __init__(self, batch_size):
        
        super(IEMC_nonphy, self).__init__()
        self.batch_size = batch_size
        # 처음에는 0으로 설정 
        # a : label 6개 중에서 어떤 것을 할 것인지에 대한 인덱스
        self.a = 0
        self.label_list = ['angry','disgust','fear','happy','neutral','sad','surprise']
       
        
    # anchor_label(0~6)을 설정해주는 버전
    def iemcl_loss_single_modal(self, label, a_m, m_1):

        # label : batch에서 y들에 대한 list 
        # a_m : 기준이 되는 modality
        # m_1, m_2 : 대상이 되는 modality 1, 2
        # anchor_label : a의 label을 가지고 있어서 anchor가 된 sample의 인덱스
        positive_score = 0
        negative_score = 0
        re_finement_score=0
        
        # label에서  label_index와 동일한 것만을 가진 index 추출 
        indices = torch.nonzero(label == self.a).squeeze(1)
        # 추출한 것에서 랜덤하게 하나 뽑아서 anchor_label로 설정
        
        # anchor의 label이 batch 내에서 유일한 경우에는 다시 뽑는다. 
        while (indices.size(0) <= 1):
            self.a = (self.a + 1) % 7
            # label에서  label_index와 동일한 것만을 가진 index 추출 
            indices = torch.nonzero(label == self.a).squeeze(1)

        # indices가 0이 아니라면 while문을 빠져 나옴
        # 추출한 것에서 랜덤하게 하나 뽑아서 anchor_label로 설정
        random_index = torch.randint(0, indices.size(0), (1,))
        self.anchor_label = indices[random_index.item()]
        
        
        
        # positive num : anchor_label과 동일한 label을 가진 샘플들의 개수 
        # negative num : anchor_label과 서로 다른 label을 가진 샘플들의 개수 
        positive_num = (label == self.a).sum().item()
        negative_num = len(label) - positive_num        
        
        
        for idx in range(label.size(0)):
            # idx가 anchor를 가리키고 있는 경우
            if self.anchor_label == idx :
                continue
           
            else:
                if label[self.anchor_label] == label[idx] :
                    positive_score += torch.dot(a_m[self.anchor_label], m_1[idx])/(torch.norm(a_m[self.anchor_label])*torch.norm(m_1[idx]))

                    re_finement_score += torch.pow(torch.dot(a_m[self.anchor_label], m_1[idx]),2)
                  
                    
                else:
                    negative_score += torch.dot(a_m[self.anchor_label], m_1[idx])/(torch.norm(a_m[self.anchor_label])*torch.norm(m_1[idx]))

                    
        if positive_num == 0:
            norm_pos_score = 0
        else:
            norm_pos_score = positive_score / positive_num

        if negative_num == 0:
            norm_neg_score = 0
        else:
            norm_neg_score = negative_score / negative_num

        if re_finement_score == 0:
            re_finement_score = 0
        else:
            re_finement_score = re_finement_score / positive_num
        
        
        
        return  2 - norm_pos_score + norm_neg_score #(norm_pos_score) / (norm_pos_score + norm_neg_score), re_finement_score

    def forward(self, label, s_t, s_a):
        
        total_score = 0
        re_finement_score = 0
        total_score1 = self.iemcl_loss_single_modal(label, s_t, s_a)#, re_finement_score1
        total_score2 = self.iemcl_loss_single_modal(label, s_a, s_t)#, re_finement_score2
        
        # 선정된 a의 label 표시 

        # 함수가 끝나기 전에 a를 한번 update
        self.a = (self.a + 1) % 7
        total_score = total_score1 + total_score2

        total_score = total_score / 2.0
   
        
        return total_score#-1 * total_score + 0.3*re_finement_score
    

class IAMC(nn.Module):

    def __init__(self, batch_size):
        super(IAMC, self).__init__()
        self.batch_size = batch_size 
        # 처음에는 0으로 설정 
        # a : label 6개 중에서 어떤 것을 할 것인지에 대한 인덱스
        self.a = 0
        self.label_list = ['angry','disgust','fear','happy','neutral','sad','surprise']

    def forward(self, label, p_t, p_v, p_a, s_t, s_v, s_a):

        # label : batch에서 y들에 대한 list 
        # a_m : 기준이 되는 modality
        # m_1, m_2 : 대상이 되는 modality 1, 2
        # anchor_label : a의 label을 가지고 있어서 anchor가 된 sample의 인덱스
        # anchor 
        # label에서  label_index와 동일한 것만을 가진 index 추출 
        indices = torch.nonzero(label == self.a).squeeze(1)
        # anchor의 label이 batch 내에서 유일한 경우에는 다시 뽑는다.
         
        while (indices.size(0)<=1):
            self.a = (self.a + 1) % 7
            # label에서  label_index와 동일한 것만을 가진 index 추출 
            indices = torch.nonzero(label == self.a).squeeze(1)

        # indices가 0이 아니라면 while문을 빠져 나옴
        # 추출한 것에서 랜덤하게 하나 뽑아서 anchor_label로 설정
        random_index = torch.randint(0, indices.size(0), (1,))
        anchor_index = indices[random_index.item()]
        
        
        anchor_label = self.label_list[self.a]

        loss_permodality = []
        
        re_finement_term_pt = []
        re_finement_term_pv = []
        re_finement_term_pa = []
        re_finement_term_ct = []
        re_finement_term_cv = []
        re_finement_term_ca = []

        re_finement_term_list = []


        pos_pt_score = []
        pos_pv_score = []
        pos_pa_score = []
        pos_ct_score = []
        pos_cv_score = []
        pos_ca_score = []
        
        neg_pt_score = []
        neg_pv_score = []
        neg_pa_score = []
        neg_ct_score = []
        neg_cv_score = []
        neg_ca_score = []
        
        
                    
        anchor_pt = p_t[anchor_index]
        anchor_pv = p_v[anchor_index]
        anchor_pa = p_a[anchor_index]
        anchor_ct = s_t[anchor_index]
        anchor_cv = s_v[anchor_index]
        anchor_ca = s_a[anchor_index]
        
        
        # Positive sample
       
        
        for sample_idx in range(label.size(0)):

           
            
            if sample_idx == anchor_index:
                continue
            
            else: 

                if self.a == label[sample_idx]:
                    
                    pos_pt_score.append(torch.dot(anchor_pt, p_t[sample_idx])/(torch.norm(anchor_pt)*torch.norm(p_t[sample_idx])))
                    re_finement_term_pt.append(torch.pow((torch.dot(anchor_pt, p_t[sample_idx])-1),2))

                    pos_pv_score.append(torch.dot(anchor_pv, p_v[sample_idx])/(torch.norm(anchor_pv)*torch.norm(p_v[sample_idx])))
                    re_finement_term_pv.append(torch.pow((torch.dot(anchor_pt, p_v[sample_idx])-1),2))

                    pos_pa_score.append(torch.dot(anchor_pa, p_a[sample_idx])/(torch.norm(anchor_pa)*torch.norm(p_a[sample_idx])))
                    re_finement_term_pa.append(torch.pow((torch.dot(anchor_pt, p_a[sample_idx])-1),2))
                    
                    pos_ct_score.append(torch.dot(anchor_ct, s_t[sample_idx])/(torch.norm(anchor_ct)*torch.norm(s_t[sample_idx])))
                    re_finement_term_ct.append(torch.pow((torch.dot(anchor_pt, s_t[sample_idx])-1),2))

                    pos_cv_score.append(torch.dot(anchor_cv, s_v[sample_idx])/(torch.norm(anchor_cv)*torch.norm(s_v[sample_idx])))
                    re_finement_term_cv.append(torch.pow((torch.dot(anchor_pt, s_v[sample_idx])-1),2))
                    
                    pos_ca_score.append(torch.dot(anchor_ca, s_a[sample_idx])/(torch.norm(anchor_ca)*torch.norm(s_a[sample_idx])))
                    re_finement_term_ca.append(torch.pow((torch.dot(anchor_pt, s_a[sample_idx])-1),2))
                    
                    
                else:
                    
                    neg_pt_score.append((torch.dot(anchor_pt, p_t[sample_idx])/(torch.norm(anchor_pt)*torch.norm(p_t[sample_idx]))))
                    neg_pv_score.append((torch.dot(anchor_pv, p_v[sample_idx])/(torch.norm(anchor_pv)*torch.norm(p_v[sample_idx]))))
                    neg_pa_score.append((torch.dot(anchor_pa, p_a[sample_idx])/(torch.norm(anchor_pa)*torch.norm(p_a[sample_idx]))))
                    
                    neg_ct_score.append((torch.dot(anchor_ct, s_t[sample_idx])/(torch.norm(anchor_ct)*torch.norm(s_t[sample_idx]))))
                    neg_cv_score.append((torch.dot(anchor_cv, s_v[sample_idx])/(torch.norm(anchor_cv)*torch.norm(s_v[sample_idx]))))
                    neg_ca_score.append((torch.dot(anchor_ca, s_a[sample_idx])/(torch.norm(anchor_ca)*torch.norm(s_a[sample_idx]))))

        
        pos_pt_score = 0 if len(pos_pt_score)==0 else (sum(pos_pt_score) / len(pos_pt_score))
        pos_pv_score = 0 if len(pos_pv_score)==0 else (sum(pos_pv_score) / len(pos_pv_score))
        pos_pa_score = 0 if len(pos_pa_score)==0 else (sum(pos_pa_score) / len(pos_pa_score))
        pos_ct_score = 0 if len(pos_ct_score)==0 else (sum(pos_ct_score) / len(pos_ct_score))
        pos_cv_score = 0 if len(pos_cv_score)==0 else (sum(pos_cv_score) / len(pos_cv_score))
        pos_ca_score = 0 if len(pos_ca_score)==0 else (sum(pos_ca_score) / len(pos_ca_score))
        neg_pt_score = 0 if len(neg_pt_score)==0 else (sum(neg_pt_score) / len(neg_pt_score))
        neg_pv_score = 0 if len(neg_pv_score)==0 else (sum(neg_pv_score) / len(neg_pv_score))
        neg_pa_score = 0 if len(neg_pa_score)==0 else (sum(neg_pa_score) / len(neg_pa_score))
        neg_ct_score = 0 if len(neg_ct_score)==0 else (sum(neg_ct_score) / len(neg_ct_score))
        neg_cv_score = 0 if len(neg_cv_score)==0 else (sum(neg_cv_score) / len(neg_cv_score))
        neg_ca_score = 0 if len(neg_ca_score)==0 else (sum(neg_ca_score) / len(neg_ca_score))

        re_finement_term_pt = 0 if len(re_finement_term_pt)==0 else (sum(re_finement_term_pt) / len(re_finement_term_pt))
        re_finement_term_pv = 0 if len(re_finement_term_pv)==0 else (sum(re_finement_term_pv) / len(re_finement_term_pv))
        re_finement_term_pa = 0 if len(re_finement_term_pa)==0 else (sum(re_finement_term_pa) / len(re_finement_term_pa))
        re_finement_term_ct = 0 if len(re_finement_term_ct)==0 else (sum(re_finement_term_ct) / len(re_finement_term_ct))
        re_finement_term_cv = 0 if len(re_finement_term_cv)==0 else (sum(re_finement_term_cv) / len(re_finement_term_cv))
        re_finement_term_ca = 0 if len(re_finement_term_ca)==0 else (sum(re_finement_term_ca) / len(re_finement_term_ca))
        
        re_finement_term_list.append(re_finement_term_pt)
        re_finement_term_list.append(re_finement_term_pv)
        re_finement_term_list.append(re_finement_term_pa)
        re_finement_term_list.append(re_finement_term_ct)
        re_finement_term_list.append(re_finement_term_cv)
        re_finement_term_list.append(re_finement_term_ca)

        

        loss_permodality.append(2-pos_pt_score+neg_pt_score) # private_t
        loss_permodality.append(2-pos_pv_score+neg_pv_score) # private_t
        loss_permodality.append(2-pos_pa_score+neg_pa_score) # private_t

        loss_permodality.append(2-pos_ct_score+neg_ct_score) # common_t
        loss_permodality.append(2-pos_cv_score+neg_cv_score) # common_t
        loss_permodality.append(2-pos_ca_score+neg_ca_score) # common_t

        
        iamcl_loss = sum(loss_permodality) / len(loss_permodality)
        re_finement_loss = sum(re_finement_term_list)/len(re_finement_term_list)

        # IEMCL과는 달리 IAMCL은 배치마다 1번만 돌아가므로 forward에서 self.a = (self.a + 1) % 7
 
        self.a = (self.a + 1) % 7

        return iamcl_loss  # -1*torch.as_tensor(iamcl_loss) #+ 0.3*re_finement_loss


class IEMC(nn.Module):

    def __init__(self, batch_size):
        
        super(IEMC, self).__init__()
        self.batch_size = batch_size
        # 처음에는 0으로 설정 
        # a : label 6개 중에서 어떤 것을 할 것인지에 대한 인덱스
        self.a = 0
        self.label_list = ['angry','disgust','fear','happy','neutral','sad','surprise']
       
        
    # anchor_label(0~6)을 설정해주는 버전
    def iemcl_loss_single_modal(self, label, a_m, m_1, m_2):

        # label : batch에서 y들에 대한 list 
        # a_m : 기준이 되는 modality
        # m_1, m_2 : 대상이 되는 modality 1, 2
        # anchor_label : a의 label을 가지고 있어서 anchor가 된 sample의 인덱스
        positive_score = 0
        negative_score = 0
        re_finement_score=0
        
        # label에서  label_index와 동일한 것만을 가진 index 추출 
        indices = torch.nonzero(label == self.a).squeeze(1)
        # 추출한 것에서 랜덤하게 하나 뽑아서 anchor_label로 설정
        
        # anchor의 label이 batch 내에서 유일한 경우에는 다시 뽑는다. 
        while (indices.size(0) <= 1):
            self.a = (self.a + 1) % 7
            # label에서  label_index와 동일한 것만을 가진 index 추출 
            indices = torch.nonzero(label == self.a).squeeze(1)

        # indices가 0이 아니라면 while문을 빠져 나옴
        # 추출한 것에서 랜덤하게 하나 뽑아서 anchor_label로 설정
        random_index = torch.randint(0, indices.size(0), (1,))
        self.anchor_label = indices[random_index.item()]
        
        
        
        # positive num : anchor_label과 동일한 label을 가진 샘플들의 개수 
        # negative num : anchor_label과 서로 다른 label을 가진 샘플들의 개수 
        positive_num = (label == self.a).sum().item()
        negative_num = len(label) - positive_num        
        
        
        for idx in range(label.size(0)):
            # idx가 anchor를 가리키고 있는 경우
            if self.anchor_label == idx :
                continue

            else:
                if label[self.anchor_label] == label[idx] :
                    positive_score += torch.dot(a_m[self.anchor_label], m_1[idx])/(torch.norm(a_m[self.anchor_label])*torch.norm(m_1[idx]))
                    positive_score += torch.dot(a_m[self.anchor_label], m_2[idx])/(torch.norm(a_m[self.anchor_label])*torch.norm(m_2[idx]))
                    re_finement_score += torch.pow(torch.dot(a_m[self.anchor_label], m_1[idx]),2)
                    re_finement_score += torch.pow(torch.dot(a_m[self.anchor_label], m_2[idx]),2)
                    
                else:
                    negative_score += torch.dot(a_m[self.anchor_label], m_1[idx])/(torch.norm(a_m[self.anchor_label])*torch.norm(m_1[idx]))
                    negative_score += torch.dot(a_m[self.anchor_label], m_2[idx])/(torch.norm(a_m[self.anchor_label])*torch.norm(m_2[idx]))
                    
        if positive_num == 0:
            norm_pos_score = 0
        else:
            norm_pos_score = positive_score / positive_num

        if negative_num == 0:
            norm_neg_score = 0
        else:
            norm_neg_score = negative_score / negative_num

        if re_finement_score == 0:
            re_finement_score = 0
        else:
            re_finement_score = re_finement_score / positive_num
        
        
        
        return  2 - norm_pos_score + norm_neg_score #(norm_pos_score) / (norm_pos_score + norm_neg_score), re_finement_score

    def forward(self, label, s_t, s_v, s_a):
        
        total_score = 0
        re_finement_score = 0
        total_score1 = self.iemcl_loss_single_modal(label, s_t, s_v, s_a)#, re_finement_score1
        total_score2 = self.iemcl_loss_single_modal(label, s_a, s_t, s_v)#, re_finement_score2
        total_score3 = self.iemcl_loss_single_modal(label, s_v, s_a, s_t)#, re_finement_score3
        # 선정된 a의 label 표시 

        # 함수가 끝나기 전에 a를 한번 update
        self.a = (self.a + 1) % 7
        total_score = total_score1 + total_score2 + total_score3
        total_score = total_score / 3
        
        return total_score#-1 * total_score + 0.3*re_finement_score
    


