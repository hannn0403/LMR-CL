# train/train_wav.py

import torch
import time
from config.config import get_config, activation_dict
from data_loaders.loader import get_loader_wav
from solvers.solver_wav import Solver

def main():
    # 설정 객체 생성 (명령행 인자 또는 기본값 사용)
    train_config = get_config(mode='train')
    test_config  = get_config(mode='test')

    # device 설정: config에 device 속성이 없다면 기본값 지정
    if not hasattr(train_config, 'device'):
        train_config.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if not hasattr(test_config, 'device'):
        test_config.device = train_config.device

    # 필요한 하이퍼파라미터 기본값 설정
    if not hasattr(train_config, 'num_classes'):
        train_config.num_classes = 7
    if not hasattr(train_config, 'dropout'):
        train_config.dropout = 0.5
    if not hasattr(train_config, 'activation'):
        # activation_dict는 config/config.py에 정의된 dict (예: {'relu': torch.nn.ReLU, ...})
        train_config.activation = activation_dict.get('relu', torch.nn.ReLU)
    if not hasattr(train_config, 'acoustic_size'):
        train_config.acoustic_size = 563472
    if not hasattr(train_config, 'n_epoch'):
        train_config.n_epoch = 3
    if not hasattr(train_config, 'patience'):
        train_config.patience = 60
    if not hasattr(train_config, 'name'):
        # 이름에 타임스탬프를 부여하여 체크포인트 파일명에 활용
        train_config.name = "train_wav_" + time.strftime("%Y%m%d_%H%M%S", time.localtime())
    if not hasattr(train_config, 'optimizer'):
        import torch.optim as optim
        train_config.optimizer = optim.Adam

    # 재현성을 위한 시드 설정
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    # WAV 데이터 로더 생성 (deploy용: finetuning=False)
    train_loader = get_loader_wav(train_config, finetuning=False, shuffle=True)
    test_loader  = get_loader_wav(test_config, finetuning=False, shuffle=False)

    # Solver 객체 생성 및 학습 수행
    solver = Solver(train_config, test_config, train_loader, test_loader, is_train=True)
    solver.build()
    solver.train()

if __name__ == '__main__':
    main()
