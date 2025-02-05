# bio_feature_extractor.py

import pandas as pd
import numpy as np
import os
import sklearn
from scipy.stats import skew, kurtosis
import glob
import warnings
import pywt
import pyeeg
import gc 
import torch
from tqdm.notebook import tqdm
from torch.optim import AdamW
from torch.nn import functional as F
warnings.filterwarnings(action='ignore')
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, classification_report, accuracy_score
from transformers import AutoTokenizer, ElectraForSequenceClassification
import torch.nn as nn

# =============================================================================
# Section 1: 텍스트 레이블 추출 (각 session 내의 txt 파일 읽기)
# =============================================================================
def make_txt_labels_20(txt_file_list, txt_file_path):
    """
    주어진 경로(txt_file_path) 내의 txt_file_list 파일들을 읽어 각 파일의
    모든 줄을 리스트로 반환한다.
    """
    return_list = []
    for txt_file in txt_file_list:
        file_path = os.path.join(txt_file_path, txt_file)
        with open(file_path, 'r', encoding='cp949') as f:
            lines = f.readlines()
        # 각 줄의 \n 제거
        lines = [line.strip() for line in lines]
        return_list.append(lines)
    return return_list

# 기본 경로 설정 (필요에 따라 수정)
BASE_DIR = os.getcwd()
KEMDY20_WAV_PATH = os.path.join(BASE_DIR, 'Data', 'KEMDy20_v1_1', 'wav')
TEXT_LABELS_DIR = os.path.join(BASE_DIR, 'Data', 'KEMDy20_v1_1', '20_text_labels_files')
ANNOT_PATH = os.path.join(BASE_DIR, 'Data', 'KEMDy20_v1_1', 'annotation')
SESSION_LABEL_WITH_CONTENT_DIR = os.path.join(BASE_DIR, 'Data', 'KEMDy20_Session_label_with_content')

# 각 session의 wav 디렉토리에서 txt 파일들을 읽어 레이블 리스트 생성
session_list = os.listdir(KEMDY20_WAV_PATH)
session_label_list = []
for session in session_list:
    session_path = os.path.join(KEMDY20_WAV_PATH, session)
    txt_file_list = [file for file in os.listdir(session_path) if file.endswith('txt')]
    txt_label_list = make_txt_labels_20(txt_file_list, session_path)
    session_label_list.append(txt_label_list)

# 텍스트 레이블 저장 디렉토리가 없으면 생성
if not os.path.exists(TEXT_LABELS_DIR):
    os.makedirs(TEXT_LABELS_DIR)

for i, session in enumerate(session_list):
    df = pd.DataFrame()
    session_path = os.path.join(KEMDY20_WAV_PATH, session)
    txt_file_list = [file[:-4] for file in os.listdir(session_path) if file.endswith('txt')]
    # 여러 파일의 라벨 리스트를 평탄화하여 하나의 리스트로 생성
    txt_file_content = sum(session_label_list[i], [])
    conversation_df = pd.DataFrame(txt_file_content, index=txt_file_list)
    df = pd.concat([df, conversation_df], axis=0)
    # index의 첫번째 값에서 session 번호 추출 (예: 'Sess01...')
    session_num = df.index[0][4:6]
    output_csv = os.path.join(TEXT_LABELS_DIR, f"Session{session_num}_text_label.csv")
    df.to_csv(output_csv, encoding="utf-8-sig")

# =============================================================================
# Section 2: 어노테이션 파일과 텍스트 레이블 병합
# =============================================================================
revised_col = ['Numb','Wav_start', 'Wav_end', 'Segment ID','Total Evaluation_Emotion', 
               'Total Evaluation_Valence', 'Total Evaluation_Arousal', 
               'Eval01F_Emotion', 'Eval01F_Valence','Eval01F_Arousal', 
               'Eval02M_Emotion', 'Eval02M_Valence','Eval02M_Arousal', 
               'Eval03M_Emotion', 'Eval03M_Valence','Eval03M_Arousal', 
               'Eval04F_Emotion', 'Eval04F_Valence','Eval04F_Arousal', 
               'Eval05M_Emotion', 'Eval05M_Valence','Eval05M_Arousal', 
               'Eval06F_Emotion', 'Eval06F_Valence','Eval06F_Arousal', 
               'Eval07F_Emotion', 'Eval07F_Valence','Eval07F_Aroual',  # 오타 수정 가능
               'Eval08M_Emotion', 'Eval08M_Valence','Eval08M_Arousal',
               'Eval09F_Emotion', 'Eval09F_Valence','Eval09F_Arousal', 
               'Eval10M_Emotion', 'Eval10M_Valence','Eval10M_Arousal',
               'text_file_name','content']
annot_file_col_list = ['Numb', 'Wav_start', 'Wav_end', 'Segment ID','Total Evaluation_Emotion', 'content']

text_label_list_sorted = sorted(os.listdir(TEXT_LABELS_DIR))
annot_list_sorted = sorted(os.listdir(ANNOT_PATH))

if not os.path.exists(SESSION_LABEL_WITH_CONTENT_DIR):
    os.makedirs(SESSION_LABEL_WITH_CONTENT_DIR)

for i in range(len(text_label_list_sorted)):
    text_label = pd.read_csv(os.path.join(TEXT_LABELS_DIR, text_label_list_sorted[i]))
    annot_file = pd.read_csv(os.path.join(ANNOT_PATH, annot_list_sorted[i]))
    annot_file = pd.merge(annot_file, text_label, left_on='Segment ID', right_on='Unnamed: 0', how='left')
    annot_file.columns = revised_col
    annot_file = annot_file.iloc[1:, :]
    annot_file = annot_file.drop('text_file_name', axis=1)
    annot_file = annot_file.loc[:, annot_file_col_list]
    session_num = annot_list_sorted[i][4:6]
    output_csv = os.path.join(SESSION_LABEL_WITH_CONTENT_DIR, f"dataset_KEMDy20_Session{session_num}.csv")
    annot_file.to_csv(output_csv, encoding="utf-8-sig")

# 전체 세션 파일 병합
annot_file_list = os.listdir(SESSION_LABEL_WITH_CONTENT_DIR)
total_df = pd.DataFrame() 
for i, file in enumerate(annot_file_list): 
    file_path = os.path.join(SESSION_LABEL_WITH_CONTENT_DIR, file)
    if i == 0: 
        total_df = pd.read_csv(file_path)
    else: 
        total_df = pd.concat([total_df, pd.read_csv(file_path)], axis=0)
mask = total_df['Total Evaluation_Emotion'].str.count(';') < 1
total_df = total_df[mask]
total_session_csv = os.path.join(SESSION_LABEL_WITH_CONTENT_DIR, "dataset_KEMDy20_total_session.csv")
total_df.to_csv(total_session_csv, encoding='utf-8-sig')

# =============================================================================
# Section 3: EDA Feature Extraction
# =============================================================================
bio = 'EDA'
print(bio)
eda_path = os.path.join(BASE_DIR, 'Data', 'KEMDy20_v1_1', bio)

def energy_wavelet(values):
    wavelet_coefficients, _ = pywt.cwt(values, 64, 'morl') 
    return np.square(np.abs(wavelet_coefficients)).sum()
    
def entropy_wavelet(values):
    wavelet_coefficients, _ = pywt.cwt(values, 64, 'morl') 
    entropy_vals = -np.square(np.abs(wavelet_coefficients)) * np.log(np.square(np.abs(wavelet_coefficients))).sum(axis=1)
    return np.mean(entropy_vals[0])
       
def rms_wavelet(values):
    wavelet_coefficients, _ = pywt.cwt(values, 64, 'morl') 
    rms_vals = np.sqrt(np.square(wavelet_coefficients).mean(axis=1))
    return rms_vals[0]
       
def energy_distribution(values):
    wavelet_coefficients, _ = pywt.cwt(values, 4, 'morl') 
    energy_dist = np.square(np.abs(wavelet_coefficients)).sum(axis=1)
    return energy_dist[0]
    
def spectral_power(values):
    return pyeeg.bin_power(values, [0.05, 0.5], 4)[0][0]

def mean_derivative(values):
    diff = np.diff(values)
    time_diff = 1  
    derivative = diff / time_diff
    return np.mean(derivative)

def skew_(values):
    return skew(values)

def kurt_(values):
    return kurtosis(values)

def EDA_feature(df):
    grouped = df.groupby(by=['subject'], as_index=False)['normalized_signal']
    df_ft_extracted = pd.concat([
        grouped.mean(),
        grouped.std()['normalized_signal'],
        grouped.min()['normalized_signal'],
        grouped.max()['normalized_signal'],
        grouped.apply(spectral_power)['normalized_signal'],
        grouped.apply(mean_derivative)['normalized_signal'],
        grouped.apply(skew_)['normalized_signal'],
        grouped.apply(kurt_)['normalized_signal'],
        grouped.apply(energy_wavelet)['normalized_signal'],
        grouped.apply(entropy_wavelet)['normalized_signal'],
        grouped.apply(rms_wavelet)['normalized_signal'],
        grouped.apply(energy_distribution)['normalized_signal']
    ], axis=1)
    df_ft_extracted.columns = ["Segment ID", "MEAN", "STD", "MIN", "MAX", "spectral_power",
                                 "mean_derivative", "skew", "kurt", "energy_wavelet",
                                 "entropy_wavelet", "rms_wavelet", "energy_distribution"]
    return df_ft_extracted

total_Session_EDA = []
for sess in range(40):
    session_num = f"0{sess+1}" if sess < 9 else str(sess+1)
    for script in range(6):
        pattern = os.path.join(eda_path, f"Session{session_num}", f"Sess{session_num}_script0{script+1}*")
        files = glob.glob(pattern)
        for file in files:
            # file 경로의 특정 부분(예: subject 명)은 실제 경로에 따라 조정 필요
            subject_nm = file[48:-4]
            with open(file, 'r') as f:
                lines = f.readlines()
            df_list = []
            df_idx = []
            general_signal = []
            for idx, line in enumerate(lines[2:]):
                general_signal.append(float(line.split(',')[0]))
                if len(line.split(',')) == 3:
                    df_list.append(line)
                    df_idx.append(idx)
            if not df_list:
                print(f"Session{session_num}_script0{script+1}_{subject_nm}_{bio} not annotated.")
            else:
                df = pd.DataFrame(df_list, columns=['value'])
                df = df['value'].str.split(',', expand=True)
                df.columns = ['value', 'date', 'subject']
                df['subject'] = df['subject'].str.replace('\n', '')
                df = df.astype({'value': 'float'})
                # 예시로, 원래 신호의 미분값(normalized) 사용 (실제 정규화 방식은 상황에 맞게 조정)
                norm_signal = sklearn.preprocessing.normalize(np.array([np.diff(np.array(general_signal))]))[0]
                df['normalized_signal'] = [norm_signal[i-1] for i in df_idx]
                extracted_df = EDA_feature(df)
                total_Session_EDA.append(extracted_df)
total_Session_EDA = pd.concat(total_Session_EDA)
total_Session_EDA_label = []
dataset_KEMDy20_total_session = pd.read_csv(total_session_csv)
for segment_id in dataset_KEMDy20_total_session['Segment ID']:
    segment = total_Session_EDA[total_Session_EDA['Segment ID'] == segment_id]
    if not segment.empty:
        segment['Total Evaluation_Emotion'] = dataset_KEMDy20_total_session.loc[
            dataset_KEMDy20_total_session['Segment ID'] == segment_id, 'Total Evaluation_Emotion'
        ].iloc[0]
    total_Session_EDA_label.append(segment)
total_Session_EDA_label = pd.concat(total_Session_EDA_label)

# =============================================================================
# Section 4: TEMP Feature Extraction
# =============================================================================
bio = 'TEMP'
print(bio)
temp_path = os.path.join(BASE_DIR, 'Data', 'KEMDy20_v1_1', bio)

def DRoftemp(values):
    return max(values) - min(values)

def mean_slope_of_temp(values):
    return np.sum(np.abs(np.diff(values))) / (len(values) - 1)

def TEMP_feature(df):
    grouped = df.groupby(by=['subject'], as_index=False)['normalized_signal']
    df_ft_extracted = pd.concat([
        grouped.mean(),
        grouped.std()['normalized_signal'],
        grouped.min()['normalized_signal'],
        grouped.max()['normalized_signal'],
        grouped.apply(DRoftemp)['normalized_signal'],
        grouped.apply(mean_slope_of_temp)['normalized_signal']
    ], axis=1)
    df_ft_extracted.columns = ["Segment ID", "MEAN", "STD", "MIN", "MAX", "DROFTEMP", "MEAN_SLOPE_OF_TEMP"]
    return df_ft_extracted

total_Session_TEMP = []
for sess in range(40):
    session_num = f"0{sess+1}" if sess < 9 else str(sess+1)
    for script in range(6):
        pattern = os.path.join(temp_path, f"Session{session_num}", f"Sess{session_num}_script0{script+1}*")
        files = glob.glob(pattern)
        for file in files:
            subject_nm = file[49:-4]
            with open(file, 'r') as f:
                lines = f.readlines()
            df_list = []
            df_idx = []
            general_signal = []
            for idx, line in enumerate(lines[2:]):
                general_signal.append(float(line.split(',')[0]))
                if len(line.split(',')) == 3:
                    df_list.append(line)
                    df_idx.append(idx)
            if not df_list:
                print(f"Session{session_num}_script0{script+1}_{subject_nm}_{bio} not annotated.")
            else:
                df = pd.DataFrame(df_list, columns=['value'])
                df = df['value'].str.split(',', expand=True)
                df.columns = ['value', 'date', 'subject']
                df['subject'] = df['subject'].str.replace('\n', '')
                df = df.astype({'value': 'float'})
                norm_signal = sklearn.preprocessing.normalize(np.array([np.array(general_signal)]))[0]
                df['normalized_signal'] = [norm_signal[i-1] for i in df_idx]
                extracted_df = TEMP_feature(df)
                total_Session_TEMP.append(extracted_df)
total_Session_TEMP = pd.concat(total_Session_TEMP)
total_Session_TEMP_label = []
for segment_id in dataset_KEMDy20_total_session['Segment ID']:
    segment = total_Session_TEMP[total_Session_TEMP['Segment ID'] == segment_id]
    if not segment.empty:
        segment['Total Evaluation_Emotion'] = dataset_KEMDy20_total_session.loc[
            dataset_KEMDy20_total_session['Segment ID'] == segment_id, 'Total Evaluation_Emotion'
        ].iloc[0]
    total_Session_TEMP_label.append(segment)
total_Session_TEMP_label = pd.concat(total_Session_TEMP_label)

# =============================================================================
# Section 5: IBI Feature Extraction
# =============================================================================
ibi_bio = 'IBI'
ibi_path = os.path.join(BASE_DIR, 'Data', 'KEMDy20_v1_1', ibi_bio)

def rms(values):
    return np.sqrt(np.sum(np.array(values)**2) / len(values))

def HR(values):
    return 60 / np.mean(values)

def LF(values):
    rr_intervals = np.array(values)
    fft_vals = np.fft.fft(rr_intervals)
    freq = np.fft.fftfreq(len(rr_intervals), d=1)
    lf_band = (freq >= 0.04) & (freq <= 0.15)
    lf_power = np.sum(np.abs(fft_vals[lf_band])**2)
    return lf_power

def RF(values):
    rr_intervals = np.array(values)
    fft_vals = np.fft.fft(rr_intervals)
    freq = np.fft.fftfreq(len(rr_intervals), d=1)
    hf_band = (freq >= 0.15) & (freq <= 0.4)
    hf_power = np.sum(np.abs(fft_vals[hf_band])**2)
    return hf_power

def IBI_feature(df):
    grouped = df.groupby(by=['subject'], as_index=False)['normalized_signal']
    df_ft_extracted = pd.concat([
        grouped.mean(),
        grouped.apply(rms)['normalized_signal'],
        grouped.apply(HR)['normalized_signal']
    ], axis=1)
    df_ft_extracted.columns = ["Segment ID", "MEAN", "RMSSD", "HR"]
    return df_ft_extracted

total_Session_IBI = []
for sess in range(40):
    session_num = f"0{sess+1}" if sess < 9 else str(sess+1)
    globals()[f"Session{sess}_IBI"] = []
    for script in range(6):
        pattern = os.path.join(ibi_path, f"Session{session_num}", f"Sess{session_num}_script0{script+1}*")
        files = glob.glob(pattern)
        for file in files:
            subject_nm = file[48:-4]
            with open(file, 'r') as f:
                lines = f.readlines()
            df_list = []
            df_idx = []
            general_signal = []
            for idx, line in enumerate(lines[1:]):
                try:
                    general_signal.append(float(line.split(',')[1]))
                except:
                    print('error:', file)
                if len(line.split(',')) == 4:
                    df_list.append(line)
                    df_idx.append(idx)
            if not df_list:
                print(f"Session{session_num}_script0{script+1}_{subject_nm}_IBI not annotated.")
            else:
                norm_signal = sklearn.preprocessing.normalize(np.array([general_signal]))[0]
                df = pd.DataFrame(df_list, columns=['value'])
                df = df['value'].str.split(',', expand=True)
                df.columns = ['dontknow', 'value', 'date', 'subject']
                df['subject'] = df['subject'].str.replace('\n', '')
                df = df.astype({'value': 'float'})
                df['normalized_signal'] = [norm_signal[i-1] for i in df_idx]
                extracted_df = IBI_feature(df)
                globals()[f"Session{sess}_IBI"].append(extracted_df)
                total_Session_IBI.append(extracted_df)
    globals()[f"Session{sess}_IBI"] = pd.concat(globals()[f"Session{sess}_IBI"])
total_Session_IBI = pd.concat(total_Session_IBI)
total_Session_IBI_label = []
for segment_id in dataset_KEMDy20_total_session['Segment ID']:
    segment = total_Session_IBI[total_Session_IBI['Segment ID'] == segment_id]
    if not segment.empty:
        segment['Total Evaluation_Emotion'] = dataset_KEMDy20_total_session.loc[
            dataset_KEMDy20_total_session['Segment ID'] == segment_id, 'Total Evaluation_Emotion'
        ].iloc[0]
    total_Session_IBI_label.append(segment)
total_Session_IBI_label = pd.concat(total_Session_IBI_label)

# =============================================================================
# Section 6: 바이오신호 특징 결합 및 저장
# =============================================================================
Normalized_IBI_feature = total_Session_IBI_label.drop(["Total Evaluation_Emotion"], axis=1)
Normalized_TEMP_feature = total_Session_TEMP_label.drop(["Total Evaluation_Emotion"], axis=1)
Normalized_TEMP_feature = Normalized_TEMP_feature.drop(["MEAN_SLOPE_OF_TEMP"], axis=1)
Normalized_EDA_feature = total_Session_EDA_label.drop(["Total Evaluation_Emotion"], axis=1)
Normalized_TEMP_feature_list = []
Normalized_EDA_feature_list = []
for segment_id in Normalized_IBI_feature['Segment ID']:
    segment_temp = Normalized_TEMP_feature[Normalized_TEMP_feature['Segment ID'] == segment_id]
    segment_eda = Normalized_EDA_feature[Normalized_EDA_feature['Segment ID'] == segment_id]
    Normalized_TEMP_feature_list.append(segment_temp)
    Normalized_EDA_feature_list.append(segment_eda)
Normalized_TEMP_feature_df = pd.concat(Normalized_TEMP_feature_list)
Normalized_EDA_feature_df = pd.concat(Normalized_EDA_feature_list)
Normalized_TEMP_feature_df = Normalized_TEMP_feature_df.drop(["Segment ID"], axis=1)
Normalized_IBI_feature_df = Normalized_IBI_feature.drop(["Segment ID"], axis=1)
Segment_ID = Normalized_EDA_feature[["Segment ID"]]
Normalized_EDA_feature_df = Normalized_EDA_feature_df.drop(["Segment ID"], axis=1)
Normalized_TEMP_feature_df.columns = ["TEMP_MEAN", "TEMP_STD", "TEMP_MIN", "TEMP_MAX", "TEMP_DROFTEMP"]
Normalized_IBI_feature_df.columns = ["IBI_MEAN", "IBI_RMSSD", "IBI_HR"]
Normalized_EDA_feature_df.columns = ["EDA_MEAN", "EDA_STD", "EDA_MIN", "EDA_MAX", "EDA_spectral_power",
                                     "EDA_mean_derivative", "EDA_skew", "EDA_kurt",
                                     "EDA_energy_wavelet", "EDA_entropy_wavelet", "EDA_rms_wavelet",
                                     "EDA_energy_distribution"]
Segment_ID = Segment_ID.reset_index(drop=True)
Normalized_EDA_feature_df = Normalized_EDA_feature_df.reset_index(drop=True)
Normalized_TEMP_feature_df = Normalized_TEMP_feature_df.reset_index(drop=True)
Normalized_IBI_feature_df = Normalized_IBI_feature_df.reset_index(drop=True)
Biosignal_feature = pd.concat([Segment_ID, Normalized_EDA_feature_df, Normalized_TEMP_feature_df, Normalized_IBI_feature_df], axis=1)
Biosignal_feature = Biosignal_feature.drop(["index"], axis=1, errors='ignore')

# 특정 세션(예: 4, 10, 15, 16, 20, 21, 23, 26)을 test로 분리
Test_list = [4, 10, 15, 16, 20, 21, 23, 26]
conditions = []
for test in Test_list:
    if test < 10:
        conditions.append(Biosignal_feature['Segment ID'].str.startswith(f'Sess0{test}'))
    else:
        conditions.append(Biosignal_feature['Segment ID'].str.startswith(f'Sess{test}'))
combined_condition = conditions[0]
for cond in conditions[1:]:
    combined_condition = combined_condition | cond
Biosignal_test = Biosignal_feature[combined_condition]
Biosignal_train = Biosignal_feature[~combined_condition]
Biosignal_test.to_csv(os.path.join(BASE_DIR, 'Data', 'Biosignal_test.csv'), index=False)
Biosignal_train.to_csv(os.path.join(BASE_DIR, 'Data', 'Biosignal_train.csv'), index=False)

text_test = dataset_KEMDy20_total_session[dataset_KEMDy20_total_session['Segment ID'].isin(Biosignal_test['Segment ID'].values)]
text_train = dataset_KEMDy20_total_session[dataset_KEMDy20_total_session['Segment ID'].isin(Biosignal_train['Segment ID'].values)]
text_test.to_csv(os.path.join(BASE_DIR, 'Data', 'text_test.csv'), index=False)
text_train.to_csv(os.path.join(BASE_DIR, 'Data', 'text_train.csv'), index=False)

# =============================================================================
# Section 7: 텍스트 분류용 Feature Extraction (NSMC Dataset 구성, F1 Loss 및 Metric)
# =============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NSMCDataset(Dataset):
    def __init__(self, csv_file):
        self.dataset = csv_file
        self.tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        row = self.dataset.iloc[idx, :].values
        text = row[1]
        y = row[0]
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=256,
            pad_to_max_length=True,
            add_special_tokens=True
        )
        input_ids = inputs['input_ids'][0]
        attention_mask = inputs['attention_mask'][0]
        return input_ids, attention_mask, y

class F1_Loss(nn.Module):
    """
    F1 Loss 구현 (원본은 Kaggle의 Michal Haltuf 버전)
    """
    def __init__(self, epsilon=1e-7):
        super().__init__()
        self.epsilon = epsilon
    def forward(self, y_pred, y_true):
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
        f1 = 2 * (precision * recall) / (precision + recall + self.epsilon)
        f1 = f1.clamp(min=self.epsilon, max=1-self.epsilon)
        return 1 - f1.mean()

def calc_metrics(y_true, y_pred):
    test_preds = y_pred  # 이미 argmax가 적용된 값이라 가정
    test_truth = y_true
    label_list = ['angry','disgust','fear','happy','neutral','sad','surprise']
    print("Confusion Matrix (pos/neg) :")
    conf_mat = confusion_matrix(test_truth, test_preds, labels=[0, 1, 2, 3, 4, 5, 6])
    conf_mat = pd.DataFrame(conf_mat, columns=label_list, index=label_list)
    print(conf_mat)
    print("Classification Report (pos/neg) :")
    print(classification_report(test_truth, test_preds, digits=5, labels=[0, 1, 2, 3, 4, 5, 6], target_names=label_list))
    print("Accuracy (pos/neg) ", accuracy_score(test_truth, test_preds))
    return accuracy_score(test_truth, test_preds)

