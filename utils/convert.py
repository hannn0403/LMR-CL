import torch

def to_gpu(x, on_cpu=False, gpu_id=0):
    """Tensor => Variable"""
    if torch.cuda.is_available() and not on_cpu:
        x = x.cuda(gpu_id)
        #x = x.to('cuda:0')
    return x

def to_cpu(x):
    """Variable => Tensor"""
    if torch.cuda.is_available():
        x = x.cpu()
    return x.data