# models/bert.py

import math
import copy
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from transformers import ElectraForSequenceClassification

####################################
# Basic Modules and Utility Functions
####################################

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x

def attention(query, key, value, dropout=None):
    """Scaled dot-product attention."""
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

def clones(module, N):
    """Produce N identical layers."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

####################################
# Gating Block
####################################

class GatingBlock(nn.Module):
    def __init__(self, feature_size):
        super(GatingBlock, self).__init__()
        self.feature_a1 = nn.Linear(feature_size * 3, feature_size // 16)
        self.feature_a2 = nn.Linear(feature_size // 16, feature_size)
        self.feature_t1 = nn.Linear(feature_size * 3, feature_size // 16)
        self.feature_t2 = nn.Linear(feature_size // 16, feature_size)
        self.feature_p1 = nn.Linear(feature_size * 3, feature_size // 16)
        self.feature_p2 = nn.Linear(feature_size // 16, feature_size)
        self.activate = nn.Sigmoid()

    def forward(self, T_a, T_t, T_p):
        # Concatenate modalities along the last dimension
        total_modal = torch.cat((T_a, T_t, T_p), dim=-1)
        # Acoustic gating
        gate_weight_a = self.feature_a1(total_modal)
        gate_weight_a = self.feature_a2(F.relu(gate_weight_a))
        gate_weight_a = self.activate(gate_weight_a)
        gate_feature_a = gate_weight_a * T_a
        # Text gating
        gate_weight_t = self.feature_t1(total_modal)
        gate_weight_t = self.feature_t2(F.relu(gate_weight_t))
        gate_weight_t = self.activate(gate_weight_t)
        gate_feature_t = gate_weight_t * T_t
        # Physiological gating
        gate_weight_p = self.feature_p1(total_modal)
        gate_weight_p = self.feature_p2(F.relu(gate_weight_p))
        gate_weight_p = self.activate(gate_weight_p)
        gate_feature_p = gate_weight_p * T_p
        # Concatenate gated features and flatten
        total_feature = torch.cat((gate_feature_a, gate_feature_t, gate_feature_p), dim=-1)
        total_feature = total_feature.view(total_feature.shape[0], -1)
        return total_feature

####################################
# Multi-Headed Attention
####################################

class MultiHeadedAttention(nn.Module):
    def __init__(self, in_dim=256, h=4, dropout=0.5):
        super(MultiHeadedAttention, self).__init__()
        assert in_dim % h == 0, "in_dim must be divisible by h"
        self.d_k = in_dim // h
        self.h = h
        self.convs = clones(nn.Linear(in_dim, in_dim), 3)
        self.linear = nn.Linear(in_dim, in_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value):
        nbatches = query.size(0)
        # Reshape and transpose for multi-head attention
        query = query.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        key = self.convs[1](key).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        value = self.convs[2](value).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        x, self.attn = attention(query, key, value, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linear(x)

####################################
# Layer Normalization and Sublayer Connection
####################################

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std  = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerOutput(nn.Module):
    """
    Residual connection followed by layer normalization.
    """
    def __init__(self, in_dim, dropout):
        super(SublayerOutput, self).__init__()
        self.norm = LayerNorm(in_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return self.norm(x + self.dropout(sublayer(x)))

####################################
# Encoder Layer, Feed Forward and Transformer Encoder (TCE)
####################################

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, in_dim, d_ff, dropout=0.5):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(in_dim, d_ff)
        self.w_2 = nn.Linear(d_ff, in_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class EncoderLayer(nn.Module):
    """
    An encoder layer with self-attention and feed-forward sublayers.
    """
    def __init__(self, in_dim, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer_output = clones(SublayerOutput(in_dim, dropout), 2)
        self.size = in_dim
        self.conv = nn.Linear(in_dim, in_dim)

    def forward(self, x_in):
        query = self.conv(x_in)
        x = self.sublayer_output[0](query, lambda x: self.self_attn(query, x_in, x_in))
        return self.sublayer_output[1](x, self.feed_forward)

class TCE(nn.Module):
    """
    Transformer Encoder: a stack of N encoder layers.
    """
    def __init__(self, layer, N):
        super(TCE, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

####################################
# Attn Module
####################################

class Attn(nn.Module):
    def __init__(self, in_dim=256, h=4, d_ff=512, N=2, dropout=0.5):
        super(Attn, self).__init__()
        attn = MultiHeadedAttention(in_dim, h, dropout)
        ff = PositionwiseFeedForward(in_dim, d_ff, dropout)
        self.tce = TCE(EncoderLayer(in_dim, deepcopy(attn), deepcopy(ff), dropout), N)
        self.conv1 = nn.Conv1d(1, 32, kernel_size=1, stride=1)
        self.conv2 = nn.Conv1d(32, 32, kernel_size=1, stride=1)
        self.fc = nn.Linear(in_dim * 32, in_dim)

    def forward(self, x):
        x = x.unsqueeze(1)  # add channel dimension
        f_h = self.conv1(x)
        f_h = F.gelu(f_h)
        f_h = self.conv2(f_h)
        f_h = F.gelu(f_h)
        encoded_features = self.tce(f_h)
        encoded_features = encoded_features.contiguous().view(encoded_features.shape[0], -1)
        encoded_features = self.fc(encoded_features)
        return encoded_features

####################################
# LMR Model (BERT 기반)
####################################

class LMR(nn.Module):
    def __init__(self, config):
        """
        config 객체에는 아래와 같은 항목이 포함되어야 합니다.
          - config.embedding_size: 텍스트 임베딩 차원
          - config.phy_size: 생리신호(phy) 차원
          - config.acoustic_size: acoustic feature 차원
          - config.num_classes: 분류할 클래스 수
          - config.dropout: dropout 비율
          - config.activation: 활성화 함수 (예: nn.ReLU, nn.Sigmoid 등; 생성자 호출형태로 사용)
          - config.use_bert: True/False (여기서는 True로 가정)
          - config.hidden_size: 공통 공간으로 매핑할 차원
          - config.device: 'cuda:0' 또는 'cpu' 등
        """
        super(LMR, self).__init__()
        self.config = config
        self.text_size = config.embedding_size
        self.phy_size = config.phy_size
        self.acoustic_size = config.acoustic_size

        self.input_sizes = [self.text_size, self.phy_size, self.acoustic_size]
        self.hidden_sizes = [int(self.text_size), int(self.phy_size), int(self.acoustic_size)]
        self.output_size = config.num_classes
        self.dropout_rate = config.dropout
        self.activation = config.activation()
        self.tanh = nn.Tanh()

        if self.config.use_bert:
            self.bertmodel = ElectraForSequenceClassification.from_pretrained(
                "monologg/koelectra-base-v3-discriminator", num_labels=config.num_classes
            ).to(config.device)
            self.bertmodel.load_state_dict(torch.load('koelectra_f1.pt', map_location=config.device))
            self.bertmodel.classifier = Identity()
        else:
            # BERT를 사용하지 않는 경우의 처리는 별도로 구현
            raise NotImplementedError("Non-BERT mode is not implemented.")

        # Phy MLP layers
        self.phymlp1 = nn.Linear(self.input_sizes[1], 150)    
        self.phymlp2 = nn.Linear(150, 200)
        self.phymlp3 = nn.Linear(200, self.hidden_sizes[1] * 16)

        # Acoustic MLP layers for wav features
        self.wav_mlp_feature1 = nn.Linear(1024, 512)
        self.wav_mlp_feature2 = nn.Linear(512, 1024)

        # Auxiliary classifiers for each modality
        self.text_auxiliary_classifer = nn.Linear(768, config.num_classes)
        self.phy_auxiliary_classifer = nn.Linear(self.hidden_sizes[1] * 16, config.num_classes)
        self.acou_auxiliary_classifer = nn.Linear(1024, config.num_classes)

        # Projection layers to map modalities to a common space
        if self.config.use_bert:
            self.project_t = nn.Sequential(
                nn.Linear(768, config.hidden_size),
                self.activation,
                LayerNorm(config.hidden_size)
            )
        else:
            self.project_t = nn.Sequential(
                nn.Linear(self.hidden_sizes[0] * 4, config.hidden_size),
                self.activation,
                LayerNorm(config.hidden_size)
            )
        self.project_p = nn.Sequential(
            nn.Linear(self.hidden_sizes[1] * 16, config.hidden_size),
            self.activation,
            LayerNorm(config.hidden_size)
        )
        self.project_a = nn.Sequential(
            nn.Linear(1024, config.hidden_size),
            self.activation,
            LayerNorm(config.hidden_size)
        )

        # Private encoders for each modality
        self.private_t = nn.Sequential(
            Attn(in_dim=config.hidden_size, h=4, d_ff=config.hidden_size * 4, N=3, dropout=0.1),
            nn.Sigmoid()
        )
        self.private_p = nn.Sequential(
            Attn(in_dim=config.hidden_size, h=4, d_ff=config.hidden_size * 4, N=3, dropout=0.1),
            nn.Sigmoid()
        )
        self.private_a = nn.Sequential(
            Attn(in_dim=config.hidden_size, h=4, d_ff=config.hidden_size * 4, N=3, dropout=0.1),
            nn.Sigmoid()
        )

        # Shared encoder for all modalities
        self.shared = nn.Sequential(
            Attn(in_dim=config.hidden_size, h=4, d_ff=config.hidden_size * 4, N=3, dropout=0.1),
            nn.Sigmoid()
        )

        # Fusion layers for final output
        self.fusion = nn.Sequential(
            nn.Linear(config.hidden_size * 6, config.hidden_size * 3),
            self.activation,
            nn.Linear(config.hidden_size * 3, self.output_size)
        )

        # Gating block layer to combine private and shared features
        self.gate_block_layer = GatingBlock(feature_size=config.hidden_size)

    def extract_features2(self, sequence, phymlp1):
        phy_feature = phymlp1(sequence)
        phy_feature = F.relu(phy_feature)
        phy_feature = self.phymlp2(phy_feature)
        phy_feature = F.relu(phy_feature)
        phy_feature = self.phymlp3(phy_feature)
        return phy_feature

    def extract_features3(self, sequence, wav_feature_layer):
        wav_feature = wav_feature_layer(sequence)
        wav_feature = F.relu(wav_feature)
        wav_feature = self.wav_mlp_feature2(wav_feature)
        return wav_feature

    def alignment(self, phy, acoustic, bert_sent, bert_sent_mask):
        if self.config.use_bert:
            # Obtain BERT output
            bert_output = self.bertmodel(bert_sent, bert_sent_mask)[0]
            # Apply masked averaging
            masked_output = torch.mul(bert_sent_mask.unsqueeze(2), bert_output)
            mask_len = torch.sum(bert_sent_mask, dim=1, keepdim=True)
            bert_output = torch.sum(masked_output, dim=1) / mask_len
            if torch.any(mask_len == 0):
                print('Warning: Some mask lengths are zero.')
            utterance_text = bert_output
        # Extract features from physio and acoustic modalities
        utterance_phy = self.extract_features2(phy, self.phymlp1)
        utterance_audio = self.extract_features3(acoustic, self.wav_mlp_feature1)
        # Auxiliary predictions
        self.text_output = self.text_auxiliary_classifer(utterance_text)
        self.phy_output = self.phy_auxiliary_classifer(utterance_phy)
        self.acou_output = self.acou_auxiliary_classifer(utterance_audio)
        # Obtain private and shared representations
        self.shared_private(utterance_text, utterance_phy, utterance_audio)
        # Stack private and shared features for each modality
        h1 = torch.stack((self.utt_private_a, self.utt_shared_a), dim=1)
        h2 = torch.stack((self.utt_private_t, self.utt_shared_t), dim=1)
        h3 = torch.stack((self.utt_private_p, self.utt_shared_p), dim=1)
        # Combine via gating block and fuse to final output
        h = self.gate_block_layer(h1, h2, h3)
        o = self.fusion(h)
        return o

    def shared_private(self, utterance_t, utterance_p, utterance_a):
        # Project each modality into common space
        self.utt_t_orig = utterance_t = self.project_t(utterance_t)
        self.utt_p_orig = utterance_p = self.project_p(utterance_p)
        self.utt_a_orig = utterance_a = self.project_a(utterance_a)
        # Obtain private representations
        self.utt_private_t = self.private_t(utterance_t)
        self.utt_private_p = self.private_p(utterance_p)
        self.utt_private_a = self.private_a(utterance_a)
        # Obtain shared representations
        self.utt_shared_t = self.shared(utterance_t)
        self.utt_shared_p = self.shared(utterance_p)
        self.utt_shared_a = self.shared(utterance_a)

    def forward(self, sentences, phy, acoustic, lengths, bert_sent, bert_sent_mask):
        # The forward pass uses alignment to produce final prediction.
        o = self.alignment(phy, acoustic, bert_sent, bert_sent_mask)
        return o
