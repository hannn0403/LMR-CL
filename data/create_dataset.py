# create_dataset.py
import os
import pickle
import numpy as np
import pandas as pd
import torchaudio
import re
from subprocess import check_call

class KemDY20:
    def __init__(self, config, modality="bert"):
        """
        Parameters:
          - config: 설정 객체 (config.dataset_dir, config.mode 등 포함)
          - modality: "bert", "wav", "wav_finetuning" 중 하나를 지정.
        """
        self.config = config
        self.modality = modality.lower()
        self.DATA_PATH = str(config.dataset_dir)
        
        # 데이터 저장 디렉토리가 없으면 생성
        if not os.path.exists(self.DATA_PATH):
            check_call(['mkdir', '-p', self.DATA_PATH])
        
        # modality에 따라 pickle 파일 이름 결정
        if self.modality == "bert":
            self.train_pickle = os.path.join(self.DATA_PATH, 'train_20.pkl')
            self.test_pickle  = os.path.join(self.DATA_PATH, 'test_20.pkl')
        elif self.modality == "wav":
            self.train_pickle = os.path.join(self.DATA_PATH, 'train_wav_20_deploy.pkl')
            self.test_pickle  = os.path.join(self.DATA_PATH, 'test_wav_20_deploy.pkl')
        elif self.modality == "wav_finetuning":
            self.train_pickle = os.path.join(self.DATA_PATH, 'train_wav_20_fintuning.pkl')
            self.test_pickle  = os.path.join(self.DATA_PATH, 'test_wav_20_fintuning.pkl')
        else:
            raise ValueError("Invalid modality specified. Use 'bert', 'wav', or 'wav_finetuning'.")
        
        # pickle 파일이 있으면 로드, 없으면 데이터셋 생성
        try:
            self.train = self.load_pickle(self.train_pickle)
            self.test  = self.load_pickle(self.test_pickle)
        except Exception as e:
            print(f"Pickle 파일 로드 실패({e}). 데이터셋을 새로 생성합니다.")
            if self.modality == "bert":
                self.build_bert_dataset()
            elif self.modality == "wav":
                self.build_wav_dataset(deploy=True)
            elif self.modality == "wav_finetuning":
                self.build_wav_dataset(deploy=False)

    def load_pickle(self, path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    def to_pickle(self, obj, path):
        with open(path, 'wb') as f:
            pickle.dump(obj, f)

    def build_bert_dataset(self):
        """
        텍스트(BERT) 및 바이오신호, acoustic feature를 사용하는 데이터셋 생성.
        원본 create_dataset_only_bert_feature_20.py 의 로직을 통합.
        """
        # CSV 파일 읽기 (train, test)
        train_data_df = pd.read_csv('./Data/text_train.csv', index_col=0)
        test_data_df  = pd.read_csv('./Data/text_test.csv', index_col=0)
        
        # 감정 레이블 매핑
        mapping = {'angry': 0, 'disgust': 1, 'disqust': 1, 'fear': 2,
                   'happy': 3, 'neutral': 4, 'sad': 5, 'surprise': 6}
        train_data_df["Total Evaluation_Emotion"] = train_data_df["Total Evaluation_Emotion"].map(mapping)
        test_data_df["Total Evaluation_Emotion"]  = test_data_df["Total Evaluation_Emotion"].map(mapping)
        train_data_df = train_data_df.astype({"Total Evaluation_Emotion": 'int'})
        test_data_df  = test_data_df.astype({"Total Evaluation_Emotion": 'int'})
        
        # 바이오신호 데이터 (biosignal)
        train_phy_df = pd.read_csv('./Data/Biosignal_train.csv', index_col=0)
        test_phy_df  = pd.read_csv('./Data/Biosignal_test.csv', index_col=0)
        
        # acoustic feature (numpy) 파일 로드
        train_acoustic = np.load('./Data/wav_feature_20_fintuning_train.npy').squeeze(1)
        test_acoustic  = np.load('./Data/wav_feature_20_fintuning_test.npy').squeeze(1)
        
        # 텍스트 (content)와 레이블
        train_labels   = train_data_df["Total Evaluation_Emotion"].values
        train_word_id  = train_data_df['content'].values
        test_labels    = test_data_df["Total Evaluation_Emotion"].values
        test_word_id   = test_data_df['content'].values
        
        # 바이오신호 데이터는 첫 번째와 마지막 컬럼을 제외한 나머지 사용
        train_phy = train_phy_df.iloc[:, 1:-1].values
        test_phy  = test_phy_df.iloc[:, 1:-1].values
        
        EPS = 1e-6
        num_drop = 0  # 처리 과정에서 삭제된 데이터 수 (여기서는 0)
        
        self.train = []
        for key in range(len(train_labels)):
            label = train_labels[key]
            _word_id = str(train_word_id[key])
            _acoustic = train_acoustic[key]
            _phy = train_phy[key]
            
            # NaN 처리
            label = np.array([np.nan_to_num(label)])[:, np.newaxis]
            _phy = np.nan_to_num(_phy)
            _acoustic = np.nan_to_num(_acoustic)
            
            words = _word_id
            phy = np.asarray(_phy)
            acoustic = np.asarray(_acoustic)
            
            # 인스턴스별 z-정규화 (phy)
            phy = np.nan_to_num((phy - phy.mean(0, keepdims=True)) /
                                (EPS + np.std(phy, axis=0, keepdims=True)))
            
            self.train.append(((words, phy, acoustic), label))
        
        self.test = []
        for key in range(len(test_labels)):
            label = test_labels[key]
            _word_id = str(test_word_id[key])
            _acoustic = test_acoustic[key]
            _phy = test_phy[key]
            
            label = np.array([np.nan_to_num(label)])[:, np.newaxis]
            _phy = np.nan_to_num(_phy)
            _acoustic = np.nan_to_num(_acoustic)
            
            words = _word_id
            phy = np.asarray(_phy)
            acoustic = np.asarray(_acoustic)
            
            phy = np.nan_to_num((phy - phy.mean(0, keepdims=True)) /
                                (EPS + np.std(phy, axis=0, keepdims=True)))
            
            self.test.append(((words, phy, acoustic), label))
        
        print(f"Total number of {num_drop} datapoints have been dropped.")
        self.to_pickle(self.train, self.train_pickle)
        self.to_pickle(self.test, self.test_pickle)

    def build_wav_dataset(self, deploy=True):
        """
        wav 및 wav_finetuning 모달리티용 데이터셋 생성.
        deploy=True이면 wav (deploy)용, False이면 wav_finetuning용 데이터셋 생성.
        원본 create_dataset_wav_20.py와 create_dataset_wav_finetuning.py의 로직 통합.
        """
        # CSV 파일 읽기 (encoding: 'utf-8-sig')
        train_data_df = pd.read_csv('./Data/text_train.csv', index_col=0, encoding='utf-8-sig')
        test_data_df  = pd.read_csv('./Data/text_test.csv', index_col=0, encoding='utf-8-sig')
        
        mapping = {'angry': 0, 'disgust': 1, 'disqust': 1, 'fear': 2,
                   'happy': 3, 'neutral': 4, 'sad': 5, 'surprise': 6}
        train_data_df["Total Evaluation_Emotion"] = train_data_df["Total Evaluation_Emotion"].map(mapping)
        test_data_df["Total Evaluation_Emotion"]  = test_data_df["Total Evaluation_Emotion"].map(mapping)
        train_data_df = train_data_df.astype({"Total Evaluation_Emotion": 'int'})
        test_data_df  = test_data_df.astype({"Total Evaluation_Emotion": 'int'})
        
        # 'Segment ID' 열을 사용하여 파일 경로 생성
        file_names_train = train_data_df['Segment ID'].values
        file_names_test  = test_data_df['Segment ID'].values
        parent_path20 = './Data/KEMDy20_v1_1/wav/Session'
        
        train_file_paths = []
        for file in file_names_train:
            seg_file = file.split('_')
            train_file_paths.append(parent_path20 + seg_file[0][-2:] + '/' + file + '.wav')
        
        test_file_paths = []
        for file in file_names_test:
            seg_file = file.split('_')
            test_file_paths.append(parent_path20 + seg_file[0][-2:] + '/' + file + '.wav')
        
        # acoustic 데이터 로드 (torchaudio 사용)
        train_speech_arrays = []
        for audio_file in train_file_paths:
            try:
                speech_array, _ = torchaudio.load(audio_file)
                # speech_array의 각 채널(또는 waveform)을 리스트에 추가
                train_speech_arrays.extend(speech_array)
            except Exception as e:
                print(f"Error loading {audio_file}: {e}")
        
        test_speech_arrays = []
        for audio_file in test_file_paths:
            try:
                speech_array, _ = torchaudio.load(audio_file)
                test_speech_arrays.extend(speech_array)
            except Exception as e:
                print(f"Error loading {audio_file}: {e}")
        
        # 레이블 추출 (데이터프레임에서 해당 'Segment ID' 기준으로)
        train_labels = train_data_df.loc[train_data_df['Segment ID'].isin(file_names_train), ["Total Evaluation_Emotion"]].values
        train_labels = train_labels.squeeze(1)
        test_labels = test_data_df.loc[test_data_df['Segment ID'].isin(file_names_test), ["Total Evaluation_Emotion"]].values
        test_labels = test_labels.squeeze(1)
        
        EPS = 1e-6
        num_drop = 0
        
        self.train = []
        for key in range(len(train_labels)):
            label = train_labels[key]
            _acoustic = train_speech_arrays[key]
            label = np.array([np.nan_to_num(label)])[:, np.newaxis]
            _acoustic = np.nan_to_num(_acoustic)
            acoustic = np.asarray(_acoustic)
            acoustic = np.nan_to_num((acoustic - acoustic.mean(0, keepdims=True)) /
                                     (EPS + np.std(acoustic, axis=0, keepdims=True)))
            self.train.append((acoustic, label))
        
        self.test = []
        for key in range(len(test_labels)):
            label = test_labels[key]
            _acoustic = test_speech_arrays[key]
            label = np.array([np.nan_to_num(label)])[:, np.newaxis]
            _acoustic = np.nan_to_num(_acoustic)
            acoustic = np.asarray(_acoustic)
            acoustic = np.nan_to_num((acoustic - acoustic.mean(0, keepdims=True)) /
                                     (EPS + np.std(acoustic, axis=0, keepdims=True)))
            self.test.append((acoustic, label))
        
        print(f"Total number of {num_drop} datapoints have been dropped.")
        self.to_pickle(self.train, self.train_pickle)
        self.to_pickle(self.test, self.test_pickle)

    def get_data(self, mode):
        if mode == "train":
            return self.train
        elif mode == "test":
            return self.test
        else:
            raise ValueError("Mode must be 'train' or 'test'")
        


"""
ex> To create and use BERT dataset 

from config.config import get_config  # config 파일에서 설정 객체를 가져온다고 가정
from create_dataset import KemDY20

config = get_config(mode='train')  # 또는 'test'
dataset = KemDY20(config, modality="bert")
train_data = dataset.get_data("train")


(비슷하게, wav / wav_finetuning의 경우 `modality="wav"` or `"wav_finetuning"`으로 호출)
"""