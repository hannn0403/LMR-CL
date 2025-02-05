import numpy as np
import random
import math

import torch
import torch.nn as nn
from torch.autograd import Function
from transformers import  Wav2Vec2ForSequenceClassification

import torch.nn.functional as F

from utils import to_gpu

import copy
from copy import deepcopy




class LMR(nn.Module):
    def __init__(self, config):
        super(LMR, self).__init__()

        self.config = config
        
        
        self.output_size = output_size = config.num_classes
        self.dropout_rate = dropout_rate = config.dropout
        self.activation = self.config.activation()
        self.tanh = nn.Tanh()
        #self.relu = nn.functional.relu()
 
        self.wav_fintuing = Wav2Vec2ForSequenceClassification.from_pretrained("kresnik/wav2vec2-large-xlsr-korean").to('cuda:0')
        self.wav_fintuing.classifier = nn.Linear(256, 7)

        

        
    def forward(self, wav_input, wav_mask):

        o = self.wav_fintuing(wav_input, attention_mask=wav_mask)
        return o
