# models/wav.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2ForSequenceClassification

class LMR(nn.Module):
    def __init__(self, config):
        """
        Parameters:
          - config: 설정 객체로, 반드시 아래 속성들이 포함되어 있어야 합니다.
              * config.num_classes: 분류할 클래스 수 (예: 7)
              * config.dropout: dropout 비율 (예: 0.5)
              * config.activation: 활성화 함수 생성자 (예: nn.ReLU)
              * config.device: 'cuda:0' 또는 'cpu' 등
        """
        super(LMR, self).__init__()
        self.config = config
        self.output_size = config.num_classes
        self.dropout_rate = config.dropout
        self.activation = config.activation()  # 예: nn.ReLU()
        self.tanh = nn.Tanh()

        # pretrained wav2vec2 모델 불러오기
        self.wav_finetuning = Wav2Vec2ForSequenceClassification.from_pretrained(
            "kresnik/wav2vec2-large-xlsr-korean",
            num_labels=config.num_classes
        )
        # 분류기(classifier) 교체: wav2vec2의 은닉 차원은 보통 256으로 가정
        self.wav_finetuning.classifier = nn.Linear(256, config.num_classes)
        
        # config.device에 맞게 모델을 이동시키는 것은 외부에서 처리하는 것을 권장합니다.
        # 예: model.to(config.device)

    def forward(self, wav_input, wav_mask):
        """
        Parameters:
          - wav_input: Wav2Vec2Processor로 전처리된 오디오 파형 텐서 (batch, sequence_length)
          - wav_mask: attention mask 텐서 (batch, sequence_length)
        
        Returns:
          - outputs: Wav2Vec2ForSequenceClassification의 출력 (로짓 등 포함)
        """
        outputs = self.wav_finetuning(wav_input, attention_mask=wav_mask)
        return outputs
