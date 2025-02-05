# utils/utils.py

import torch
import time
import functools
import torch.nn as nn
import torch.nn.functional as F

def to_gpu(x, gpu_id=0):
    """
    주어진 tensor를 GPU가 사용 가능하면 지정한 gpu_id로 이동시킵니다.
    """
    return x.to(f'cuda:{gpu_id}') if torch.cuda.is_available() else x

def time_desc_decorator(desc):
    """
    데코레이터: 함수 실행 시간을 측정하여 desc와 함께 출력합니다.
    
    사용 예:
        @time_desc_decorator("My Function")
        def my_function(...):
            ...
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start
            print(f"{desc} took {elapsed:.2f} sec")
            return result
        return wrapper
    return decorator

class DiffLoss(nn.Module):
    """
    두 텐서 간의 차이를 제곱하여 평균을 내는 Loss.
    """
    def __init__(self):
        super(DiffLoss, self).__init__()
    
    def forward(self, private, shared):
        return torch.mean((private - shared) ** 2)

class CMD(nn.Module):
    """
    Central Moment Discrepancy (CMD) loss.
    주어진 n_moments 만큼의 모멘트 차이를 절대값 평균으로 계산합니다.
    
    사용 예:
        loss = CMD()(source, target, n_moments=3)
    """
    def __init__(self, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        super(CMD, self).__init__()
        self.kernel_mul = kernel_mul
        self.kernel_num = kernel_num
        self.fix_sigma = fix_sigma
    
    def forward(self, source, target, n_moments):
        loss = 0.0
        for i in range(1, n_moments + 1):
            source_moment = torch.mean(source ** i, dim=0)
            target_moment = torch.mean(target ** i, dim=0)
            loss += torch.mean(torch.abs(source_moment - target_moment))
        return loss

class IAMC(nn.Module):
    """
    Intra-class Adaptive Moment Alignment (IAMC) loss.
    (여기서는 간단한 L2 norm 차이를 이용한 더미 구현을 제공합니다.)
    """
    def __init__(self, feature_dim):
        super(IAMC, self).__init__()
        self.feature_dim = feature_dim
    
    def forward(self, y, private_t, private_p, private_a, shared_t, shared_p, shared_a):
        # 각 모달리티의 private와 shared 간 L2 norm의 평균을 계산
        loss = (torch.norm(private_t - shared_t, p=2) +
                torch.norm(private_p - shared_p, p=2) +
                torch.norm(private_a - shared_a, p=2)) / 3.0
        return loss

class IEMC(nn.Module):
    """
    Inter-class Euclidean Moment Alignment (IEMC) loss.
    (여기서는 각 shared 표현의 분산의 평균을 계산하는 간단한 더미 구현입니다.)
    """
    def __init__(self, feature_dim):
        super(IEMC, self).__init__()
        self.feature_dim = feature_dim
    
    def forward(self, y, shared_t, shared_p, shared_a):
        loss = (torch.var(shared_t) + torch.var(shared_p) + torch.var(shared_a)) / 3.0
        return loss

class F1_Loss(nn.Module):
    """
    F1 Loss 구현.
    모델의 예측값과 실제 레이블로부터 F1 score를 계산하고, 이를 1 - F1로 반환하여 최소화합니다.
    
    참고:
      - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    """
    def __init__(self, epsilon=1e-7):
        super(F1_Loss, self).__init__()
        self.epsilon = epsilon
    
    def forward(self, y_pred, y_true):
        # y_pred: (batch_size, num_classes)
        # y_true: (batch_size,) - 클래스 인덱스
        assert y_pred.ndim == 2, "y_pred must be of shape (batch_size, num_classes)"
        assert y_true.ndim == 1, "y_true must be of shape (batch_size,)"
        y_true_onehot = F.one_hot(y_true, num_classes=y_pred.size(1)).to(torch.float32)
        y_pred_soft = F.softmax(y_pred, dim=1)
        
        tp = (y_true_onehot * y_pred_soft).sum(dim=0).to(torch.float32)
        tn = ((1 - y_true_onehot) * (1 - y_pred_soft)).sum(dim=0).to(torch.float32)
        fp = ((1 - y_true_onehot) * y_pred_soft).sum(dim=0).to(torch.float32)
        fn = (y_true_onehot * (1 - y_pred_soft)).sum(dim=0).to(torch.float32)
        
        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)
        
        f1 = 2 * (precision * recall) / (precision + recall + self.epsilon)
        f1 = f1.clamp(min=self.epsilon, max=1 - self.epsilon)
        return 1 - f1.mean()
