import pandas as pd
import numpy as np
import os
import sklearn
import pandas as pd
import numpy as np
import os
from scipy.stats import skew
import glob
from scipy.stats import kurtosis


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


def make_txt_labels_20(txt_file_list, txt_file_path):
    return_list = []
    for txt_file in txt_file_list:
        # 해당 conversation에 저장되어 있는 txt 파일들을 읽는다. 
        f = open(txt_file_path + '/' + txt_file, 'r', encoding='cp949')
        lines = f.readlines()
        # 각 줄의 \n을 지운다. 
        for i in range(len(lines)):
            lines[i] = lines[i].strip()
        return_list.append(lines)
        f.close()
    return return_list

path_20 = './Data/KEMDy20_v1_1/wav/' # 이 부분은 해당 파일들이 있는 절대 경로 입력 
session_list = os.listdir(path_20)
# session_list

session_label_list = []

for session in session_list:
    
    session_path = path_20 + session
    # print(session_path)
    txt_file_list = [file for file in os.listdir(session_path) if file.endswith('txt')]
    txt_label_list = make_txt_labels_20(txt_file_list, session_path)
    session_label_list.append(txt_label_list)

if not os.path.exists('./Data/KEMDy20_v1_1/20_text_labels_files'):
    os.makedirs('./Data/KEMDy20_v1_1/20_text_labels_files')

for i, session in enumerate(session_list):
    df = pd.DataFrame()
    session_path = path_20 + session
    txt_file_list = [file[:-4] for file in os.listdir(session_path) if file.endswith('txt')]
    txt_file_content = sum(session_label_list[i], [])
    conversation_df = pd.DataFrame(txt_file_content, index = txt_file_list)
    df = pd.concat([df, conversation_df], axis=0)
    df.to_csv(f"./Data/KEMDy20_v1_1/20_text_labels_files/Session{df.index[0][4:6]}_text_label.csv", encoding="utf-8-sig")


revised_col = ['Numb','Wav_start', 'Wav_end', 'Segment ID','Total Evaluation_Emotion', 'Total Evaluation_Valence', 'Total Evaluation_Arousal', 
               'Eval01F_Emotion', 'Eval01F_Valence','Eval01F_Arousal', 'Eval02M_Emotion', 'Eval02M_Valence','Eval02M_Arousal', 
               'Eval03M_Emotion', 'Eval03M_Valence','Eval03M_Arousal', 'Eval04F_Emotion', 'Eval04F_Valence','Eval04F_Arousal', 
               'Eval05M_Emotion', 'Eval05M_Valence','Eval05M_Arousal', 'Eval06F_Emotion', 'Eval06F_Valence','Eval06F_Arousal', 
               'Eval07F_Emotion', 'Eval07F_Valence','Eval07F_Arousal', 'Eval08M_Emotion', 'Eval08M_Valence','Eval08M_Arousal',
               'Eval09F_Emotion', 'Eval09F_Valence','Eval09F_Arousal', 'Eval10M_Emotion', 'Eval10M_Valence','Eval10M_Arousal',
               'text_file_name','content']


text_label_path = './Data/KEMDy20_v1_1/20_text_labels_files/'
annot_path = './Data/KEMDy20_v1_1/annotation/'

text_label_list = os.listdir(text_label_path)
annot_list = os.listdir(annot_path)

annot_file_col_list = ['Numb', 'Wav_start', 'Wav_end', 'Segment ID','Total Evaluation_Emotion', 'content']

# Directory 생성 
if not os.path.exists('./Data/KEMDy20_Session_label_with_content/'):
    os.makedirs('./Data/KEMDy20_Session_label_with_content/')

text_label_list = sorted(text_label_list)
annot_list = sorted(annot_list)
# ['Sess01_eval.csv', 'Sess02_eval.csv', ....
for i in range(len(text_label_list)):

    text_label = pd.read_csv(text_label_path + text_label_list[i])
    annot_file = pd.read_csv(annot_path + annot_list[i])
    annot_file = pd.merge(annot_file, text_label, left_on='Segment ID', right_on='Unnamed: 0', how='left')
    annot_file.columns = revised_col
    annot_file = annot_file.iloc[1:, :]
    annot_file = annot_file.drop('text_file_name', axis=1)
    annot_file = annot_file.loc[:, annot_file_col_list]
    annot_file.to_csv(f'./Data/KEMDy20_Session_label_with_content/dataset_KEMDy20_Session{annot_list[i][4:6]}.csv', encoding="utf-8-sig")


# Total Session file 
annot_file_list = os.listdir('./Data/KEMDy20_Session_label_with_content/')

total_df = pd.DataFrame() 
for i in range(len(annot_file_list)): 
    if i == 0: 
        total_df = pd.read_csv('./Data/KEMDy20_Session_label_with_content/'+annot_file_list[i])
    else: 
        total_df = pd.concat([total_df, pd.read_csv('./Data/KEMDy20_Session_label_with_content/' + annot_file_list[i])], axis=0)
mask = total_df['Total Evaluation_Emotion'].str.count(';')<1
total_df = total_df[mask]
total_df.to_csv('./Data/KEMDy20_Session_label_with_content/dataset_KEMDy20_total_session.csv', encoding='utf-8-sig')


warnings.filterwarnings("ignore")
session_length = []

dataset_KEMDy20_total_session = pd.read_csv('Data/KEMDy20_Session_label_with_content/dataset_KEMDy20_total_session.csv')

badtrial_drop = ['Sess01_script05_User002M','Sess01_script06_User002M','Sess03_script04_User005M','Sess04_script06_User007M','Sess05_script04_User009M',
                 'Sess05_script06_User009M','Sess18_script03_User036M','Sess36_script03_User072F']

# EDA
bio = 'EDA'
print(bio)
path = 'Data/KEMDy20_v1_1/{}/'.format(bio)

# EDA feature Extraction

def energy_wavelet(values):
    wavelet_coefficients, _ = pywt.cwt(values, 64, 'morl') 
    return np.square(np.abs(wavelet_coefficients)).sum()
    
def entropy_wavelet(values):
    wavelet_coefficients, _ = pywt.cwt(values, 64, 'morl') 
    entropy_wavelet = -np.square(np.abs(wavelet_coefficients)) * np.log(np.square(np.abs(wavelet_coefficients))).sum(axis=1)
    return np.mean(entropy_wavelet[0])
       
def rms_wavelet(values):
    wavelet_coefficients, _ = pywt.cwt(values, 64, 'morl') 
    rms_wavelet = np.sqrt(np.square(wavelet_coefficients).mean(axis=1))
    return rms_wavelet[0]
       
def energy_distribution(values):
    wavelet_coefficients, _ = pywt.cwt(values, 4, 'morl') 
    energy_distribution = np.square(np.abs(wavelet_coefficients)).sum(axis=1)
    return energy_distribution[0]#, energy_wavelet, np.mean(entropy_wavelet[0]), rms_wavelet[0] # Energy distribution, energyWavelet, entropyWavelet, rmsWavelet
    
def spectral_power(values):
    return pyeeg.bin_power(values, [0.05, 0.5], 4)[0][0]

def mean_energy(values):
    freq_domain = np.abs(fft(values))
    return np.mean(freq_domain)

def mean_derivative(values):
    diff = np.diff(values)  # 이전 데이터와의 차이 계산
    time_diff = 1  # 시간 간격이 1이라고 가정
    derivative = diff / time_diff  # 변화율 계산
    mean_derivative = np.mean(derivative)  # 평균 변화율 계산
    return mean_derivative

def skew_(values):
    return skew(values)

def kurt_(values):
    return kurtosis(values)

def derivative(values):
    diff = np.diff(values)
    time_diff = 1 
    derivative = diff / time_diff
    return derivative

def EDA_feature(df):
    #print('==============',df['normalized_signal'].astype(float))
    df = df.groupby(by=['subject'],as_index=False)['normalized_signal']
    #df = df.groupby(by=['subject'])['normalized_signal']
    #print("===========",df.std()['normalized_signal'])
    
    
    
    
    df_ft_extracted = pd.concat([df.mean(), df.std()['normalized_signal'], df.min()['normalized_signal'],
                                 df.max()['normalized_signal'],
                                df.apply(spectral_power)['normalized_signal'], 
                                 df.apply(mean_derivative)['normalized_signal'],
                                df.apply(skew_)['normalized_signal'], 
                                 df.apply(kurt_)['normalized_signal'], df.apply(energy_wavelet)['normalized_signal'], df.apply(entropy_wavelet)['normalized_signal'],df.apply(rms_wavelet)['normalized_signal'],
                                 df.apply(energy_distribution)['normalized_signal']],axis=1)
    
    df_ft_extracted.columns=["Segment ID", "MEAN", "STD", "MIN", "MAX", "spectral_power", "mean_derivative", "skew", "kurt", "energy_wavelet", "entropy_wavelet", "rms_wavelet", "energy_distribution"]
    
    return df_ft_extracted

total_Session_EDA = []

for sess in range(40): # i is session num
    session_num = '0'+str(sess+1) if sess < 9 else sess+1

    for script in range(6): # q is script num 
        script_file_path = path + 'Session{}/Sess{}_script0{}*'.format(session_num, session_num, script+1)
        globals()['Session{}_script{}'.format(session_num, script+1)] = glob.glob(script_file_path)
        
        for file in globals()['Session{}_script{}'.format(session_num, script+1)]:
            subject_nm = file[48:-4]
            with open(file) as f:
                lines = f.readlines()
                
            df = []
            df_idx = [] #contain the segment index
            general_signal=[]
            
            for line in lines[2:]:
                general_signal.append(float(line.split(',')[:1][0]))
            
            cumulative_signal = np.array(general_signal)
            original_signal = np.diff(cumulative_signal)
            general_signal = sklearn.preprocessing.normalize(np.array([original_signal]))
            
            for idx, line in enumerate(lines[2:]):
                if len(line.split(',')) == 3:
                    df.append(line)
                    df_idx.append(idx)
            
            df = pd.DataFrame(df)

            if df.empty:
                print('Session{}_script{}_{}_{}'.format(session_num, script+1,subject_nm, bio)) # not annotated file

            else:
                
                df.columns=['value']
                df = df.value.str.split(',')
                df = df.apply(lambda x: pd.Series(x))
                df.columns=['value', 'date', 'subject']
                df['subject'] = df['subject'].replace('\n', '', regex=True)
                df = df.astype({'value':'float'})
                
                df['normalized_signal'] = [general_signal[0][i-1] for i in df_idx]
                
               # df.columns = ['normalized_signal']
                
                globals()['Session{}_script0{}_{}_{}_df'.format(session_num, script+1, subject_nm, bio)] = df
                #df= pd.DataFrame(df.groupby('subject').apply(lambda x : list(x['normalized_signal']))).reset_index()
                #df.columns=['subject','normalized_signal'] 
                
                globals()['Session{}_script0{}_{}_EDA_ft_extracted_df'.format(session_num, script+1,subject_nm)] = EDA_feature(globals()['Session{}_script0{}_{}_{}_df'.format(session_num, script+1, subject_nm, bio)])#df
                total_Session_EDA.append(globals()['Session{}_script0{}_{}_EDA_ft_extracted_df'.format(session_num, script+1,subject_nm)])
                
                
total_Session_EDA = pd.concat(total_Session_EDA)
#total_Session_EDA.columns = ['Segment ID', 'EDA']
total_Session_EDA_label=[]
for segment_id in dataset_KEMDy20_total_session['Segment ID']:
    segment = total_Session_EDA[total_Session_EDA['Segment ID'] == segment_id]
    
    if segment.empty:
        #print(segment_id)
        pass
    else:
        segment['Total Evaluation_Emotion'] = list(dataset_KEMDy20_total_session[dataset_KEMDy20_total_session['Segment ID'] == list(segment['Segment ID'])[0]]['Total Evaluation_Emotion'])[0]
    total_Session_EDA_label.append(segment)
    
    
total_Session_EDA_label = pd.concat(total_Session_EDA_label)

# TEMP

bio='TEMP'
print(bio)
path = 'Data/KEMDy20_v1_1/{}/'.format(bio)

# TEMP feature Extraction

def DRoftemp(values):
    return max(values)-min(values)

def mean_slope_of_temp(value):
    return np.sum(np.abs(value[1:] - value[:-1])) / (len(value) - 1)

def TEMP_feature(df):
    
    df_ft_extracted = pd.concat([df.groupby(by=['subject'],as_index=False)['normalized_signal'].mean(), df.groupby(by=['subject'],as_index=False)['normalized_signal'].std()['normalized_signal'], df.groupby(by=['subject'],as_index=False)['normalized_signal'].min()['normalized_signal'],
                                 df.groupby(by=['subject'],as_index=False)['normalized_signal'].max()['normalized_signal'], df.groupby(by=['subject'],as_index=False)['normalized_signal'].apply(DRoftemp)['normalized_signal'], 
                                 df.groupby(by=['subject'],as_index=False)['normalized_signal'].apply(mean_slope_of_temp)['normalized_signal']], axis=1) #.columns=["MEAN","RMSSD","HR"]
    
    df_ft_extracted.columns=["Segment ID", "MEAN", "STD", "MIN", "MAX", "DROFTEMP", "MEAN_SLOPE_OF_TEMP"]
    
    return df_ft_extracted



total_Session_TEMP = []
for sess in range(40): # i is session num
    session_num = '0'+str(sess+1) if sess < 9 else sess+1
    
    for script in range(6): # q is script num 
        script_file_path = path + 'Session{}/Sess{}_script0{}*'.format(session_num, session_num, script+1)
        globals()['Session{}_script{}'.format(session_num, script+1)] = glob.glob(script_file_path)
        
        
        for file in globals()['Session{}_script{}'.format(session_num, script+1)]:
            subject_nm = file[49:-4]
            
            with open(file) as f:
                lines = f.readlines()
                
            df = []
            df_idx = [] #contain the segment index
            general_signal=[]
            
            for line in lines[2:]:
                general_signal.append(float(line.split(',')[:1][0]))
            
            #general_signal = pd.DataFrame(general_signal, columns=[subject_nm])
            general_signal = sklearn.preprocessing.normalize(np.array([general_signal]))
            
            
            for idx, line in enumerate(lines[2:]):
                if len(line.split(',')) == 3:
                    df.append(line)
                    df_idx.append(idx)
                    
            df = pd.DataFrame(df)
            
            if df.empty:
                print('Session{}_script{}_{}_{}'.format(session_num, script+1,subject_nm, bio)) # not annotated file
                
            else:
                
                df.columns=['value']
                df = df.value.str.split(',')
                df = df.apply(lambda x: pd.Series(x))
                df.columns=['value', 'date', 'subject']
                df['subject'] = df['subject'].replace('\n', '', regex=True)
                df = df.astype({'value':'float'})
                df['normalized_signal'] = [general_signal[0][i-1] for i in df_idx]
                globals()['Session{}_script0{}_{}_{}_df'.format(session_num, script+1,subject_nm, bio)] = df
                globals()['Session{}_script0{}_{}_TEMP_ft_extracted_df'.format(session_num, script+1,subject_nm)] = TEMP_feature(globals()['Session{}_script0{}_{}_{}_df'.format(session_num, script+1,subject_nm, bio)])
               # globals()['Session{}_{}'.format(sess, bio)].append(globals()['Session{}_script0{}_{}_{}_ft_extracted_df'.format(session_num, script+1,subject_nm, bio)])
                total_Session_TEMP.append(globals()['Session{}_script0{}_{}_TEMP_ft_extracted_df'.format(session_num, script+1,subject_nm)])
                
total_Session_TEMP = pd.concat(total_Session_TEMP)

total_Session_TEMP_label=[]
for segment_id in dataset_KEMDy20_total_session['Segment ID']:
    segment = total_Session_TEMP[total_Session_TEMP['Segment ID'] == segment_id]
    
    if segment.empty:
        #print(segment_id)
        pass
    else:
        segment['Total Evaluation_Emotion'] = list(dataset_KEMDy20_total_session[dataset_KEMDy20_total_session['Segment ID'] == list(segment['Segment ID'])[0]]['Total Evaluation_Emotion'])[0]
    total_Session_TEMP_label.append(segment)
total_Session_TEMP_label = pd.concat(total_Session_TEMP_label)

total_Session_IBI_label=[]

def rms(values):
    return np.sqrt(sum(values**2)/len(values))


def HR(values):
    return 60/np.mean(values)


def LF(values):
    rr_intervals = np.array(values)
    fft = np.fft.fft(rr_intervals)
    freq = np.fft.fftfreq(len(rr_intervals), d=1)

    lf_band = (freq >= 0.04) & (freq <= 0.15)
    lf_power = np.sum(np.abs(fft[lf_band])**2)
    return lf_power


def RF(values):
    rr_intervals = np.array(values)
    fft = np.fft.fft(rr_intervals)
    freq = np.fft.fftfreq(len(rr_intervals), d=1)

    hf_band = (freq >= 0.15) & (freq <= 0.4)
    hf_power = np.sum(np.abs(fft[hf_band])**2)
    return hf_power


def LFRFratio(values):
    return lf_power / hf_power
    
# SD1 and SD2 contains NaN value

def SD1(values):
    return np.std(np.diff(values)) / np.sqrt(2)


def SD2(values):
    return np.sqrt(0.5* np.mean(np.diff(values)**2 - np.mean(np.diff(values))**2))


def IBI_feature(df):
    df_ft_extracted = pd.concat([df.groupby(by=['subject'],as_index=False)['normalized_signal'].mean(), df.groupby(by=['subject'],as_index=False)['normalized_signal'].apply(rms)['normalized_signal'], df.groupby(by=['subject'],as_index=False)['normalized_signal'].apply(HR)['normalized_signal']], axis=1) #.columns=["MEAN","RMSSD","HR"]
    df_ft_extracted.columns=["Segment ID", "MEAN", "RMSSD", "HR"]
    return df_ft_extracted


path = 'Data/KEMDy20_v1_1/{}/'.format('IBI')


# Feature dataframe extractor
total_Session_IBI = []

for sess in range(40): # i is session num
    session_num = '0'+str(sess+1) if sess < 9 else sess+1
    globals()['Session{}_IBI'.format(sess)] = []
    
    for script in range(6): # q is script num 
        script_file_path = path + 'Session{}/Sess{}_script0{}*'.format(session_num, session_num, script+1)
        globals()['Session{}_script{}_IBI'.format(session_num, script+1)] = glob.glob(script_file_path)
        
        for file in globals()['Session{}_script{}_IBI'.format(session_num, script+1)]:
            subject_nm = file[48:-4]
            
            with open(file) as f:
                lines = f.readlines() 
                
            df = []
            df_idx = [] #contain the segment index
            general_signal=[]
            
            for line in lines[1:]:
                try:
                    general_signal.append(float(line.split(',')[1]))
                except:
                    print('error: ', file)
            
            
            for idx, line in enumerate(lines[1:]):
                
                if len(line.split(',')) == 4:
                    
                    df.append(line)
                    df_idx.append(idx)
                    
            df = pd.DataFrame(df)
            
            if df.empty:
                print('Session{}_script{}_{}_{}'.format(session_num, script+1,subject_nm, 'IBI')) # not annotated file

            else:
                general_signal = sklearn.preprocessing.normalize(np.array([general_signal]))
                df.columns=['value']
                df = df.value.str.split(',')
                df = df.apply(lambda x: pd.Series(x))
                df.columns=['dontknow','value', 'date', 'subject']
                df['subject'] = df['subject'].replace('\n', '', regex=True)
                df = df.astype({'value':'float'})
                df['normalized_signal'] = [general_signal[0][i-1] for i in df_idx]
                globals()['Session{}_script0{}_{}_IBI_df'.format(session_num, script+1,subject_nm)] = df
                globals()['Session{}_script0{}_{}_IBI_ft_extracted_df'.format(session_num, script+1,subject_nm)] = IBI_feature(globals()['Session{}_script0{}_{}_IBI_df'.format(session_num, script+1,subject_nm)])
                globals()['Session{}_IBI'.format(sess)].append(globals()['Session{}_script0{}_{}_IBI_ft_extracted_df'.format(session_num, script+1,subject_nm)])
                total_Session_IBI.append(globals()['Session{}_script0{}_{}_IBI_ft_extracted_df'.format(session_num, script+1,subject_nm)])
                
                
    globals()['Session{}_IBI'.format(sess)] = pd.concat(globals()['Session{}_IBI'.format(sess)])
    
total_Session_IBI = pd.concat(total_Session_IBI)

for segment_id in dataset_KEMDy20_total_session['Segment ID']:
    segment = total_Session_IBI[total_Session_IBI['Segment ID'] == segment_id]
    
    if segment.empty:
        pass
    else:
        segment['Total Evaluation_Emotion'] = list(dataset_KEMDy20_total_session[dataset_KEMDy20_total_session['Segment ID'] == list(segment['Segment ID'])[0]]['Total Evaluation_Emotion'])[0]
    total_Session_IBI_label.append(segment)
total_Session_IBI_label = pd.concat(total_Session_IBI_label)


Normalized_IBI_feature = total_Session_IBI_label
Normalized_IBI_feature = Normalized_IBI_feature.drop(["Total Evaluation_Emotion"], axis=1)
Normalized_TEMP_feature = total_Session_TEMP_label.drop(["Total Evaluation_Emotion"], axis=1)
Normalized_TEMP_feature= Normalized_TEMP_feature.drop(["MEAN_SLOPE_OF_TEMP"], axis=1)
Normalized_EDA_feature = total_Session_EDA_label.drop(["Total Evaluation_Emotion"], axis=1)
Normalized_TEMP_feature_ = []
Normalized_EDA_feature_ = []

for segment_id in Normalized_IBI_feature['Segment ID']:
    segment_temp = Normalized_TEMP_feature[Normalized_TEMP_feature['Segment ID'] == segment_id]
    segment_eda = Normalized_EDA_feature[Normalized_EDA_feature['Segment ID'] == segment_id]
    Normalized_TEMP_feature_.append(segment_temp)
    Normalized_EDA_feature_.append(segment_eda)
   # print(segment)

Normalized_TEMP_feature_ = pd.concat(Normalized_TEMP_feature_)
Normalized_EDA_feature_ = pd.concat(Normalized_EDA_feature_)

Normalized_TEMP_feature_ = Normalized_TEMP_feature_.drop(["Segment ID"], axis=1)
Normalized_IBI_feature_ = Normalized_IBI_feature.drop(["Segment ID"], axis=1)
Segment_ID = Normalized_EDA_feature_["Segment ID"]
Normalized_EDA_feature_ = Normalized_EDA_feature_.drop(["Segment ID"], axis=1)
Normalized_TEMP_feature_.columns=["TEMP_MEAN", "TEMP_STD","TEMP_MIN","TEMP_MAX","TEMP_DROFTEMP"]
Normalized_IBI_feature_.columns=["IBI_MEAN", "IBI_RMSSD","IBI_HR"]
Normalized_EDA_feature_.columns = ["EDA_MEAN", "EDA_STD", "EDA_MIN", "EDA_MAX", "EDA_spectral_power","EDA_mean_derivative", "EDA_skew", "EDA_kurt", "EDA_energy_wavelet", "EDA_entropy_wavelet", "EDA_rms_wavelet", "EDA_energy_distribution"]

Segment_ID= Segment_ID.reset_index()
Normalized_EDA_feature_ = Normalized_EDA_feature_.reset_index()
Normalized_TEMP_feature_ = Normalized_TEMP_feature_.reset_index()
Normalized_IBI_feature_ = Normalized_IBI_feature_.reset_index()
Biosignal_feature = pd.concat([Segment_ID, Normalized_EDA_feature_, Normalized_TEMP_feature_, Normalized_IBI_feature_],axis=1)
Biosignal_feature = Biosignal_feature.drop(["index"], axis=1)


Test_list = [4, 10, 15, 16, 20, 21, 23, 26]

for test in Test_list:
    if test < 10:
        globals()[f'condition0{test}'] = Biosignal_feature['Segment ID'].str.startswith(f'Sess0{test}')
    else:
        globals()[f'condition{test}'] = Biosignal_feature['Segment ID'].str.startswith(f'Sess{test}')
        
Biosignal_test = Biosignal_feature[condition04 | condition10 | condition15 | condition16 | condition20 | condition21 | condition23 | condition26]
Biosignal_train = Biosignal_feature[~(condition04 | condition10 | condition15 | condition16 | condition20 | condition21 | condition23 | condition26)]
Biosignal_test.to_csv('./Data/Biosignal_test.csv')
Biosignal_train.to_csv('./Data/Biosignal_train.csv')

text_test = dataset_KEMDy20_total_session[dataset_KEMDy20_total_session['Segment ID'].isin(Biosignal_test['Segment ID'].values)]
text_train = dataset_KEMDy20_total_session[dataset_KEMDy20_total_session['Segment ID'].isin(Biosignal_train['Segment ID'].values)]

text_test.to_csv('./Data/text_test.csv')
text_train.to_csv('./Data/text_train.csv')

# ======================== Text feature =======================

# GPU 사용
device = torch.device("cuda")



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
    '''Calculate F1 score. Can work with gpu tensors
    
    The original implmentation is written by Michal Haltuf on Kaggle.
    
    Returns
    -------
    torch.Tensor
        `ndim` == 1. epsilon <= val <= 1
    
    Reference
    ---------
    - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6
    - http://www.ryanzhang.info/python/writing-your-own-loss-function-module-for-pytorch/
    '''
    def __init__(self, epsilon=1e-7):
        super().__init__()
        self.epsilon = epsilon
        
    def forward(self, y_pred, y_true,):
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

        f1 = 2* (precision*recall) / (precision + recall + self.epsilon)
        f1 = f1.clamp(min=self.epsilon, max=1-self.epsilon)
        return 1 - f1.mean()
    


def calc_metrics(y_true, y_pred):
    
        # test_preds = np.argmax(y_pred, 1)
        test_preds = y_pred
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


column_list = ['Segment ID', 'content','Total Evaluation_Emotion']

label_list = ['angry','disgust','fear','happy','neutral','sad','surprise']

train_data = pd.read_csv('./Data/text_train.csv', encoding='utf-8-sig')
# test_data = pd.read_csv('./Data/text_test.csv', encoding='cp949') 

train_data = train_data.loc[:, column_list]
# test_data = test_data.loc[:, column_list]
train_data.columns = ['Segment ID', 'content','label']
# test_data.columns = ['Segment ID', 'content','label']


train_data = train_data[['label','content']]
# test_data = test_data[['label','content']]

train_data.columns = ['emotion','content']
# test_data.columns = ['emotion','content']


train_data.loc[(train_data['emotion'] == 'angry'), 'emotion'] = 0 # angry -> 0
train_data.loc[(train_data['emotion'] == 'disgust'), 'emotion'] = 1 # disqust -> 1
train_data.loc[(train_data['emotion'] == 'disqust'), 'emotion'] = 1 # disqust -> 1
train_data.loc[(train_data['emotion'] == 'fear'), 'emotion'] = 2 # fear -> 2
train_data.loc[(train_data['emotion'] == 'happy'), 'emotion'] = 3 # happy -> 3
train_data.loc[(train_data['emotion'] == 'neutral'), 'emotion'] =4 # neutral -> 4
train_data.loc[(train_data['emotion'] == 'sad'), 'emotion'] = 5 # sad -> 5
train_data.loc[(train_data['emotion'] == 'surprise'), 'emotion'] = 6 # surprise -> 6


'''
test_data.loc[(test_data['emotion'] == 'angry'), 'emotion'] = 0 # angry -> 0
test_data.loc[(test_data['emotion'] == 'disgust'), 'emotion'] = 1 # disqust -> 1
test_data.loc[(test_data['emotion'] == 'fear'), 'emotion'] = 2 # fear -> 2
test_data.loc[(test_data['emotion'] == 'happy'), 'emotion'] = 3 # happy -> 3
test_data.loc[(test_data['emotion'] == 'neutral'), 'emotion'] =4 # neutral -> 4
test_data.loc[(test_data['emotion'] == 'sad'), 'emotion'] = 5 # sad -> 5
test_data.loc[(test_data['emotion'] == 'surprise'), 'emotion'] = 6 # surprise -> 6
'''

emotion_list = ['angry', 'disgust','fear','happy','neutral','sad','surprise']


# train_data, val_data = train_test_split(train_data, test_size=0.2, stratify=train_data['emotion'], random_state=42)
train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)
# Test Session is 4, 10, 15, 16, 20, 21, 23, 26
dataset_train = train_data
dataset_val = val_data
# dataset_test = test_data

#NSMCDataset 클래스 이용, TensorDataset으로 만들어주기
train_dataset = NSMCDataset(dataset_train)
val_dataset = NSMCDataset(dataset_val)
# test_dataset = NSMCDataset(dataset_test)


model = ElectraForSequenceClassification.from_pretrained("monologg/koelectra-base-v3-discriminator", num_labels=7).to(device)

# epochs = 30
# epochs = 150
epochs = 10
batch_size = 16
optimizer = AdamW(model.parameters(), lr=5e-6)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)


losses = []
accuracies = []
best_f1 = 0.0

# 추가 부분
label_list = ['angry','disgust','fear','happy','neutral','sad','surprise']

for i in range(epochs):
    print(f'====================== EPOCH {i} ==============================')
    total_loss = 0.0
    correct = 0
    total = 0
    batches = 0
    incorrect_prediction_gt = pd.DataFrame()
    correct_prediction_gt = pd.DataFrame()
    total_prediction = pd.DataFrame()
    model.train()

    for input_ids_batch, attention_masks_batch, y_batch in tqdm(train_loader):
        optimizer.zero_grad()
        y_batch = y_batch.to(device)
        y_pred = model(input_ids_batch.to(device), attention_mask=attention_masks_batch.to(device))[0]
        
        f1_loss = F1_Loss().to(device)
        loss = f1_loss(y_pred, y_batch)
        
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        _, predicted = torch.max(y_pred, 1)


        y_batch= y_batch.to('cpu')
        predicted = predicted.to('cpu')
        
        # Record incorrect samples
        incorrect_pred = []
        incorrect_gt = []
        correct_pred = []
        correct_gt = []
        for j in range(len(y_batch)):
            if predicted[j] != y_batch[j]:
                incorrect_pred.append(predicted[j])
                incorrect_gt.append(y_batch[j])
            else:
                correct_pred.append(predicted[j])
                correct_gt.append(y_batch[j])


        incorrect_dict = {'pred' : incorrect_pred, 'gt' : incorrect_gt}
        correct_dict = {'pred' : correct_pred, 'gt' : correct_gt}
        # Creating df of incorrect smaples in the batch
        incorr_df = pd.DataFrame(incorrect_dict)
        corr_df = pd.DataFrame(correct_dict)
        # concatenation
        incorrect_prediction_gt = pd.concat([incorrect_prediction_gt, incorr_df], axis=0)
        correct_prediction_gt = pd.concat([correct_prediction_gt, corr_df], axis=0)
        total_prediction = pd.concat([incorrect_prediction_gt, correct_prediction_gt], axis=0)
        
        total_prediction['pred'] = total_prediction['pred'].apply(lambda x : x.item())
        total_prediction['gt'] = total_prediction['gt'].apply(lambda x : x.item())
        
        
        
        y_batch = y_batch.to(device)
        predicted = predicted.to(device)        
        
        correct += (predicted == y_batch).sum()
        total += len(y_batch)

        batches += 1
        if batches % 50 == 0:
            print("Batch Loss:", total_loss, "Accuracy:", correct.float() / total)
  
    calc_metrics(total_prediction['gt'], total_prediction['pred'])
    
    conf_mat = confusion_matrix(total_prediction['gt'], total_prediction['pred'], labels=[0, 1, 2, 3, 4, 5, 6])
    conf_mat = pd.DataFrame(conf_mat, columns=label_list, index=label_list)
    
    
    losses.append(total_loss)
    accuracies.append(correct.float() / total)
    print("Train Loss:", total_loss, "Accuracy:", correct.float() / total)
    
    model.eval()

    # test_correct = 0
    # test_total = 0 
    val_correct = 0
    val_total = 0


    incorrect_prediction_gt = pd.DataFrame()
    correct_prediction_gt = pd.DataFrame()
    total_prediction = pd.DataFrame()

    for input_ids_batch, attention_masks_batch, y_batch in tqdm(val_loader):
        y_batch = y_batch.to(device)
        y_pred = model(input_ids_batch.to(device), attention_mask = attention_masks_batch.to(device))[0]
        _, predicted = torch.max(y_pred, 1)

        y_batch = y_batch.to('cpu')
        predicted = predicted.to('cpu')
        
        # Record incorrect samples
        incorrect_pred = []
        incorrect_gt = []
        correct_pred = []
        correct_gt = []
        for j in range(len(y_batch)):
            if predicted[j] != y_batch[j]:
                incorrect_pred.append(predicted[j])
                incorrect_gt.append(y_batch[j])
            else:
                correct_pred.append(predicted[j])
                correct_gt.append(y_batch[j])
        
        incorrect_dict = {'pred' : incorrect_pred, 'gt' : incorrect_gt}
        correct_dict = {'pred' : correct_pred, 'gt' : correct_gt}
        # Creating df of incorrect smaples in the batch
        incorr_df = pd.DataFrame(incorrect_dict)
        corr_df = pd.DataFrame(correct_dict)
        # concatenation
        incorrect_prediction_gt = pd.concat([incorrect_prediction_gt, incorr_df], axis=0)
        correct_prediction_gt = pd.concat([correct_prediction_gt, corr_df], axis=0)
        total_prediction = pd.concat([incorrect_prediction_gt, correct_prediction_gt], axis=0)
        
        total_prediction['pred'] = total_prediction['pred'].apply(lambda x : x.item())
        total_prediction['gt'] = total_prediction['gt'].apply(lambda x : x.item())

        

        y_batch = y_batch.to(device)
        predicted = predicted.to(device)
        val_correct += (predicted == y_batch).sum()
        val_total += len(y_batch)
        
    calc_metrics(total_prediction['gt'], total_prediction['pred'])
    
    
    conf_mat = confusion_matrix(total_prediction['gt'], total_prediction['pred'], labels=[0, 1, 2, 3, 4, 5, 6])
    conf_mat = pd.DataFrame(conf_mat, columns=label_list, index=label_list)
    
    
    f1 = f1_score(total_prediction['gt'], total_prediction['pred'], average='macro')
    if f1 > best_f1:
        best_f1 = f1
        # Save model 
        torch.save(model.state_dict(), f"./koelectra_f1.pt")
    
    
    gc.collect()
    torch.cuda.empty_cache()
    