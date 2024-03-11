import numpy as np
import math

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F

from transformers import ElectraForSequenceClassification

import copy
from copy import deepcopy






class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

def attention(query, key, value, dropout=None):
    "Implementation of Scaled dot product attention"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class GatingBlock(nn.Module):
    def __init__(self, feature_size):
        super(GatingBlock, self).__init__()

        self.feature_a1 = nn.Linear(feature_size*3, feature_size//16)
        self.feature_a2 = nn.Linear(feature_size//16, feature_size)
        self.feature_t1 = nn.Linear(feature_size*3, feature_size//16)
        self.feature_t2 = nn.Linear(feature_size//16, feature_size)
        self.feature_p1 = nn.Linear(feature_size*3, feature_size//16)
        self.feature_p2 = nn.Linear(feature_size//16, feature_size)
        self.activate = nn.Sigmoid()


    def forward(self, T_a, T_t, T_p):
        total_modal = torch.cat((T_a, T_t, T_p), dim=-1) # batch, 2, config.hidden_size*3
        gate_weight_a = self.feature_a1(total_modal)
        gate_weight_a = self.feature_a2(F.relu(gate_weight_a))
        gate_weight_a = self.activate(gate_weight_a)
        gate_feature_a = gate_weight_a * T_a

        gate_weight_t = self.feature_t1(total_modal)
        gate_weight_t = self.feature_t2(F.relu(gate_weight_t))
        gate_weight_t = self.activate(gate_weight_t)
        gate_feature_t = gate_weight_t * T_t

        gate_weight_p = self.feature_p1(total_modal)
        gate_weight_p = self.feature_p2(F.relu(gate_weight_p))
        gate_weight_p = self.activate(gate_weight_p)
        gate_feature_p = gate_weight_p * T_p # batch, 2, confi.hidden_size


        total_feature = torch.cat((gate_feature_a, gate_feature_t, gate_feature_p), dim=-1) # batch, 2, config.hidden_size*3
        total_feature = total_feature.view(total_feature.shape[0], -1)

        return total_feature




class MultiHeadedAttention(nn.Module):
    def __init__(self, in_dim=256 , h=4, dropout=0.5):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert in_dim % h == 0
        self.d_k = in_dim // h
        self.h = h

        self.convs = clones(nn.Linear(in_dim, in_dim), 3)
        self.linear = nn.Linear(in_dim, in_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value):
        "Implements Multi-head attention"
        nbatches = query.size(0)

        query = query.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        key   = self.convs[1](key).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        value = self.convs[2](value).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)

        x, self.attn = attention(query, key, value, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)

        return self.linear(x)


class LayerNorm(nn.Module):
    "Construct a layer normalization module."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerOutput(nn.Module):
    '''
    A residual connection followed by a layer norm.
    '''

    def __init__(self, in_dim, dropout):
        super(SublayerOutput, self).__init__()
        self.norm = LayerNorm(in_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return self.norm(x + self.dropout(sublayer(x)))


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class TCE(nn.Module):
    '''
    Transformer Encoder
    It is a stack of N layers.
    '''

    def __init__(self, layer, N):
        super(TCE, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class EncoderLayer(nn.Module):
    '''
    An encoder layer
    Made up of self-attention and a feed forward layer.
    Each of these sublayers have residual and layer norm, implemented by SublayerOutput.
    '''
    def __init__(self, in_dim, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer_output = clones(SublayerOutput(in_dim, dropout), 2)
        self.size = in_dim
        self.conv = nn.Linear(in_dim, in_dim)


    def forward(self, x_in):
        "Transformer Encoder"
        query = self.conv(x_in)
        x = self.sublayer_output[0](query, lambda x: self.self_attn(query, x_in, x_in))  # Encoder self-attention
        return self.sublayer_output[1](x, self.feed_forward)


class PositionwiseFeedForward(nn.Module):
    "Positionwise feed-forward network."

    def __init__(self, in_dim, d_ff, dropout=0.5):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(in_dim, d_ff)
        self.w_2 = nn.Linear(d_ff, in_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        "Implements FFN equation."
        return self.w_2(self.dropout(F.relu(self.w_1(x))))



class Attn(nn.Module):
    def __init__(self, in_dim=256, h=4, d_ff = 512, N=2, dropout=0.5):
        super(Attn, self).__init__()

        attn = MultiHeadedAttention(in_dim, h, dropout)
        ff = PositionwiseFeedForward(in_dim, d_ff, dropout)
        self.tce = TCE(EncoderLayer(in_dim, deepcopy(attn), deepcopy(ff), dropout), N)
        self.conv1 = nn.Conv1d(1, 32, kernel_size=1, stride=1)
        self.conv2 = nn.Conv1d(32, 32, kernel_size=1, stride=1) 

        self.fc = nn.Linear(in_dim * 32, in_dim)

    def forward(self, x):

        x = x.unsqueeze(1)
        f_h = self.conv1(x)
        f_h = nn.functional.gelu(f_h)
        f_h = self.conv2(f_h)
        f_h = nn.functional.gelu(f_h)

        encoded_features = self.tce(f_h)
        encoded_features = encoded_features.contiguous().view(encoded_features.shape[0], -1)
        encoded_features = self.fc(encoded_features)
        return encoded_features



# let's define a simple model that can deal with multimodal variable length sequence
class LMR(nn.Module):
    def __init__(self, config):
        super(LMR, self).__init__()

        self.config = config
        self.text_size = config.embedding_size
        # size 확인 필요
        self.phy_size = config.phy_size
        self.acoustic_size = config.acoustic_size


        self.input_sizes = input_sizes = [self.text_size, self.phy_size, self.acoustic_size]
        self.hidden_sizes = hidden_sizes = [int(self.text_size), int(self.phy_size), int(self.acoustic_size)]
        self.output_size = output_size = config.num_classes
        self.dropout_rate = dropout_rate = config.dropout
        self.activation = self.config.activation()
        self.tanh = nn.Tanh()
        #self.relu = nn.functional.relu()
        
  
        if self.config.use_bert:

            # Initializing a BERT bert-base-uncased style configuration
            self.bertmodel = ElectraForSequenceClassification.from_pretrained("monologg/koelectra-base-v3-discriminator", num_labels=7).to('cuda:0')
            self.bertmodel.load_state_dict(torch.load('koelectra_f1.pt', map_location='cuda:0'))
            self.bertmodel.classifier = Identity()

        else:
            self.embed = nn.Embedding(len(config.word2id), input_sizes[0])
            self.trnn1 = rnn(input_sizes[0], hidden_sizes[0], bidirectional=True)
            self.trnn2 = rnn(2*hidden_sizes[0], hidden_sizes[0], bidirectional=True)
        
        
        self.phymlp1 = nn.Linear(input_sizes[1], 150)    
        self.phymlp2 = nn.Linear(150, 200)
        self.phymlp3 = nn.Linear(200, hidden_sizes[1]*16)
        
        self.wav_mlp_feature1 = nn.Linear(1024, 512)
        self.wav_mlp_feature2 = nn.Linear(512, 1024)

    
        # auxiliary classifier
        self.text_auxiliary_classifer = nn.Linear(768, 7)
        self.phy_auxiliary_classifer = nn.Linear(hidden_sizes[1]*16, 7)
        self.acou_auxiliary_classifer = nn.Linear(1024, 7)



        ##########################################
        # mapping modalities to same sized space
        ##########################################
        if self.config.use_bert:
            self.project_t = nn.Sequential()
            self.project_t.add_module('project_t', nn.Linear(in_features=768, out_features=config.hidden_size))
            self.project_t.add_module('project_t_activation', self.activation)
            self.project_t.add_module('project_t_layer_norm', LayerNorm(config.hidden_size))
        else:
            self.project_t = nn.Sequential()
            self.project_t.add_module('project_t', nn.Linear(in_features=hidden_sizes[0]*4, out_features=config.hidden_size))
            self.project_t.add_module('project_t_activation', self.activation)
            self.project_t.add_module('project_t_layer_norm', LayerNorm(config.hidden_size))

        self.project_p = nn.Sequential()
        self.project_p.add_module('project_p', nn.Linear(in_features=hidden_sizes[1]*16, out_features=config.hidden_size))
        self.project_p.add_module('project_p_activation', self.activation)
        self.project_p.add_module('project_p_layer_norm', LayerNorm(config.hidden_size))

        self.project_a = nn.Sequential()
        self.project_a.add_module('project_a', nn.Linear(in_features=1024, out_features=config.hidden_size))
        self.project_a.add_module('project_a_activation', self.activation)
        self.project_a.add_module('project_a_layer_norm', LayerNorm(config.hidden_size))


        ##########################################
        # private encoders
        ##########################################
        self.private_t = nn.Sequential()
        self.private_t.add_module('private_t_1' , Attn(in_dim=config.hidden_size, h=4, d_ff = config.hidden_size*4, N=3, dropout=0.1))
        self.private_t.add_module('private_t_activation_1', nn.Sigmoid())
        
        self.private_p = nn.Sequential()
        self.private_p.add_module('private_p_1', Attn(in_dim=config.hidden_size, h=4, d_ff = config.hidden_size*4, N=3, dropout=0.1))
        self.private_p.add_module('private_p_activation_1', nn.Sigmoid())
        
        self.private_a = nn.Sequential()
        self.private_a.add_module('private_a_1', Attn(in_dim=config.hidden_size, h=4, d_ff = config.hidden_size*4, N=3, dropout=0.1))
        self.private_a.add_module('private_a_activation_1', nn.Sigmoid())
        

        ##########################################
        # shared encoder
        ##########################################
        self.shared = nn.Sequential()
        self.shared.add_module('shared_1', Attn(in_dim=config.hidden_size, h=4, d_ff = config.hidden_size*4, N=3, dropout=0.1))
        self.shared.add_module('shared_activation_1', nn.Sigmoid())


        self.fusion = nn.Sequential()
        self.fusion.add_module('fusion_layer_1', nn.Linear(in_features=self.config.hidden_size*6, out_features=self.config.hidden_size*3))
        self.fusion.add_module('fusion_layer_1_activation', self.activation)
        self.fusion.add_module('fusion_layer_2', nn.Linear(in_features=self.config.hidden_size*3, out_features= output_size))
        
        self.gate_block_layer = GatingBlock(feature_size = self.config.hidden_size)

        

        
    def extract_features(self, sequence, lengths, rnn1, rnn2, layer_norm):
        packed_sequence = pack_padded_sequence(sequence, lengths)

        if self.config.rnncell == "lstm":
            packed_h1, (final_h1, _) = rnn1(packed_sequence)
        else:
            packed_h1, final_h1 = rnn1(packed_sequence)

        padded_h1, _ = pad_packed_sequence(packed_h1)
        normed_h1 = layer_norm(padded_h1)
        packed_normed_h1 = pack_padded_sequence(normed_h1, lengths)

        if self.config.rnncell == "lstm":
            _, (final_h2, _) = rnn2(packed_normed_h1)
        else:
            _, final_h2 = rnn2(packed_normed_h1)

        return final_h1, final_h2
        
    def extract_features2(self, sequence, phymlp1):
        
        phy_feature = phymlp1(sequence)
        phy_feature = nn.functional.relu(phy_feature)
        phy_feature = self.phymlp2(phy_feature)
        phy_feature = nn.functional.relu(phy_feature)
        phy_feature = self.phymlp3(phy_feature)      
        return phy_feature

    def extract_features3(self, sequence, wav_feature_layer):
        
        wav_feature = wav_feature_layer(sequence)
        wav_feature = nn.functional.relu(wav_feature)
        wav_feature = self.wav_mlp_feature2(wav_feature)
      
        return wav_feature
        
        

    def alignment(self, phy, acoustic, bert_sent, bert_sent_mask):
        
        if self.config.use_bert:
            bert_output = self.bertmodel(bert_sent, bert_sent_mask)      

            bert_output = bert_output[0]
            
            # masked mean
            masked_output = torch.mul(bert_sent_mask.unsqueeze(2), bert_output)
            mask_len = torch.sum(bert_sent_mask, dim=1, keepdim=True)  
            bert_output = torch.sum(masked_output, dim=1, keepdim=False) / mask_len
            if 0 in mask_len:
                print('=======================mask_len is zero')

            utterance_text = bert_output
            

        utterance_phy = self.extract_features2(phy, self.phymlp1)
   
        utterance_audio = self.extract_features3(acoustic, self.wav_mlp_feature1)

        # auxiliary classifier
        #utterance_text, utterance_audio, utterance_phy
        self.text_output = self.text_auxiliary_classifer(utterance_text)
        self.phy_output = self.phy_auxiliary_classifer(utterance_phy)
        self.acou_output = self.acou_auxiliary_classifer(utterance_audio)

        # Shared-private encoders
        self.shared_private(utterance_text, utterance_phy, utterance_audio)

        

        # Gating Block
        h1 = torch.stack((self.utt_private_a, self.utt_shared_a), dim=1)
        h2 = torch.stack((self.utt_private_t, self.utt_shared_t), dim=1)
        h3 = torch.stack((self.utt_private_p, self.utt_shared_p), dim=1)
        h = self.gate_block_layer(h1, h2, h3) #b, 2* config*3
        o = self.fusion(h)

        return o

    def shared_private(self, utterance_t, utterance_p, utterance_a):
        
        # Projecting to same sized space
        self.utt_t_orig = utterance_t = self.project_t(utterance_t)
        self.utt_p_orig = utterance_p = self.project_p(utterance_p)
        self.utt_a_orig = utterance_a = self.project_a(utterance_a)
        

        # Private-shared components
        self.utt_private_t = self.private_t(utterance_t)
        self.utt_private_p = self.private_p(utterance_p)
        self.utt_private_a = self.private_a(utterance_a)

        self.utt_shared_t = self.shared(utterance_t)
        self.utt_shared_p = self.shared(utterance_p)
        self.utt_shared_a = self.shared(utterance_a)


    def forward(self, sentences, phy, acoustic, lengths, bert_sent, bert_sent_mask):

        o = self.alignment(phy, acoustic, bert_sent, bert_sent_mask)
        return o
