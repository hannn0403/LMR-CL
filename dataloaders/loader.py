# loader.py
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, Wav2Vec2Processor

# data/create_dataset.py 에 정의된 KemDY20 클래스를 import합니다.
from data.create_dataset import KemDY20

###############################################
# BERT 기반 데이터셋 및 DataLoader
###############################################

class MSADatasetBERT(Dataset):
    """
    BERT 기반 데이터셋.
    각 샘플은 ((text, phy, acoustic), label) 형태로 구성되어 있으며,
    텍스트 부분은 이후 BERT 토크나이저를 이용해 인코딩됩니다.
    """
    def __init__(self, config):
        # KemDY20 클래스에서 modality="bert"로 데이터셋 생성
        self.dataset = KemDY20(config, modality="bert")
        self.data = self.dataset.get_data(config.mode)
        self.len = len(self.data)
        # BERT 토크나이저 (monologg/koelectra-base-v3-discriminator)
        self.tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
        # 예시: config에 추가 속성 설정 (필요에 따라 조정)
        config.phy_size = 19
        if config.mode == 'train':
            config.acoustic_size = 563472
        elif config.mode == 'test':
            config.acoustic_size = 563472

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len

def collate_fn_bert(batch):
    """
    배치 내 각 샘플은 ((text, phy, acoustic), label) 구조입니다.
    - texts: 각 샘플의 텍스트(str)
    - phy: numpy array → FloatTensor
    - acoustic: numpy array → FloatTensor
    - label: numpy array → Tensor (concatenation)
    - BERT 토크나이저를 이용해 input_ids, attention_mask 생성
    - lengths: 단순히 각 샘플 길이를 1로 처리 (추후 RNN 사용 시 필요)
    """
    import torch

    # 레이블들을 텐서로 합칩니다.
    labels = torch.cat([torch.from_numpy(sample[1]) for sample in batch], dim=0)

    # 텍스트, 바이오(phy), acoustic feature 추출
    texts = [str(sample[0][0]) for sample in batch]
    phy = torch.FloatTensor(np.array([sample[0][1] for sample in batch]))
    acoustic = torch.FloatTensor(np.array([sample[0][2] for sample in batch]))

    # BERT 토크나이저를 이용하여 texts 인코딩 (padding은 max_length 기준)
    tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
    bert_details = tokenizer(
        texts,
        max_length=256,
        return_tensors='pt',
        truncation=True,
        padding='max_length'
    )
    bert_sentences = bert_details['input_ids']       # (batch_size, 256)
    bert_att_mask = bert_details['attention_mask']     # (batch_size, 256)

    # 길이 정보 (모든 샘플의 길이를 1로 설정; 필요 시 수정)
    lengths = torch.LongTensor([1 for _ in batch])

    return texts, phy, acoustic, labels, lengths, bert_sentences, bert_att_mask

def get_loader_bert(config, shuffle=True):
    """
    BERT 기반 데이터 로더 생성 함수.
    config.mode ('train' 또는 'test')에 맞게 데이터셋을 불러오고,
    지정한 배치 크기(config.batch_size)와 shuffle 옵션에 따라 DataLoader를 리턴합니다.
    """
    dataset = MSADatasetBERT(config)
    config.data_len = len(dataset)
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=shuffle, collate_fn=collate_fn_bert)


###############################################
# WAV 기반 데이터셋 및 DataLoader (deploy / finetuning)
###############################################

class MSADatasetWAV(Dataset):
    """
    WAV 기반 데이터셋.
    modality 인자에 따라 "wav" (deploy용) 또는 "wav_finetuning" (파인튜닝용) 데이터셋을 생성합니다.
    각 샘플은 (acoustic, label) 형태로 구성됩니다.
    """
    def __init__(self, config, finetuning=False):
        modality = "wav_finetuning" if finetuning else "wav"
        self.dataset = KemDY20(config, modality=modality)
        self.data = self.dataset.get_data(config.mode)
        self.len = len(self.data)
        # Wav2Vec2Processor from pretrained "kresnik/wav2vec2-large-xlsr-korean"
        self.wav_processor = Wav2Vec2Processor.from_pretrained("kresnik/wav2vec2-large-xlsr-korean")
        if config.mode == 'train':
            config.acoustic_size = 563472
        elif config.mode == 'test':
            config.acoustic_size = 563472

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len

def collate_fn_wav(batch):
    """
    WAV 데이터의 배치 처리 함수.
    각 샘플은 (acoustic, label) 형태입니다.
    Wav2Vec2Processor를 이용해 오디오 파형을 토큰화 및 패딩합니다.
    """
    import torch

    processor = Wav2Vec2Processor.from_pretrained("kresnik/wav2vec2-large-xlsr-korean")
    wav_features_list = []
    for sample in batch:
        # sample[0]는 오디오 파형(numpy array 또는 Tensor)
        wav_features = processor(
            sample[0],
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )
        wav_features_list.append(wav_features)
    # 입력 특징들을 리스트로 구성 후, pad 함수를 통해 일괄 패딩
    input_features = [{'input_values': wf['input_values'][0]} for wf in wav_features_list]
    wav_batch = processor.pad(input_features, return_tensors='pt')
    wav_input = wav_batch['input_values']      # (batch, sequence_length)
    wav_att_mask = wav_batch['attention_mask']   # (batch, sequence_length)
    labels = torch.cat([torch.from_numpy(sample[1]) for sample in batch], dim=0)
    return wav_input, wav_att_mask, labels

def get_loader_wav(config, finetuning=False, shuffle=True):
    """
    WAV 기반 DataLoader 생성 함수.
    finetuning 인자가 True이면 파인튜닝용 데이터셋("wav_finetuning")을, False이면 deploy용("wav")을 사용합니다.
    배치 크기는 파인튜닝 시 4, 그렇지 않으면 config.batch_size를 사용합니다.
    """
    dataset = MSADatasetWAV(config, finetuning=finetuning)
    config.data_len = len(dataset)
    batch_size = 4 if finetuning else config.batch_size
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn_wav)