import sys
import os
import re
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook
from subprocess import check_call, CalledProcessError

import torch
import torch.nn as nn 
import torchaudio


def to_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)



class KemDY20:
    def __init__(self, config): 

        
        DATA_PATH = str(config.dataset_dir)
        CACHE_PATH = DATA_PATH + '/embedding_and_mapping.pt'

        # If cached data if already exists
        try:
            self.train = load_pickle(DATA_PATH + '/train_wav_20_fintuning.pkl')
            self.test = load_pickle(DATA_PATH + '/test_wav_20_fintuning.pkl')

        except:


            # create folders for storing the data
            if not os.path.exists(DATA_PATH):
                check_call(' '.join(['mkdir', '-p', DATA_PATH]), shell=True) 

            train_data_df = pd.read_csv('./Data/text_train.csv', index_col=0, encoding = 'utf-8-sig')

            train_data_df.loc[train_data_df["Total Evaluation_Emotion"] == 'angry', "Total Evaluation_Emotion"] = 0
            train_data_df.loc[train_data_df["Total Evaluation_Emotion"] == 'disgust', "Total Evaluation_Emotion"] = 1
            train_data_df.loc[train_data_df["Total Evaluation_Emotion"] == 'disqust', "Total Evaluation_Emotion"] = 1
            train_data_df.loc[train_data_df["Total Evaluation_Emotion"] == 'fear', "Total Evaluation_Emotion"] = 2
            train_data_df.loc[train_data_df["Total Evaluation_Emotion"] == 'happy', "Total Evaluation_Emotion"] = 3
            train_data_df.loc[train_data_df["Total Evaluation_Emotion"] == 'neutral', "Total Evaluation_Emotion"] = 4
            train_data_df.loc[train_data_df["Total Evaluation_Emotion"] == 'sad', "Total Evaluation_Emotion"] = 5
            train_data_df.loc[train_data_df["Total Evaluation_Emotion"] == 'surprise', "Total Evaluation_Emotion"] = 6
            train_data_df =train_data_df.astype({"Total Evaluation_Emotion":'int'})
            
            file_names = train_data_df['Segment ID'].values

            parent_path20 = './Data/KEMDy20_v1_1/wav/Session'


            file_name_pst = []
            file_path_list = []

            for file in file_names:
                seg_file = file.split('_')
                file_name_pst.append(file)
                file_path_list.append(parent_path20+seg_file[0][-2:]+'/'+file+'.wav')
            
            train_speech_arrays = []
            for audio_file in file_path_list:
                speech_array,_ = torchaudio.load(audio_file)
                train_speech_arrays.extend(speech_array)


            train_labels = train_data_df["Total Evaluation_Emotion"].values

            train_acoustic = train_speech_arrays
            
            
            #Test
            test_data_df = pd.read_csv('./Data/text_test.csv', index_col=0, encoding = 'utf-8-sig')
            test_data_df.loc[test_data_df["Total Evaluation_Emotion"] == 'angry', "Total Evaluation_Emotion"] = 0
            test_data_df.loc[test_data_df["Total Evaluation_Emotion"] == 'disgust', "Total Evaluation_Emotion"] = 1
            test_data_df.loc[test_data_df["Total Evaluation_Emotion"] == 'disqust', "Total Evaluation_Emotion"] = 1
            test_data_df.loc[test_data_df["Total Evaluation_Emotion"] == 'fear', "Total Evaluation_Emotion"] = 2
            test_data_df.loc[test_data_df["Total Evaluation_Emotion"] == 'happy', "Total Evaluation_Emotion"] = 3
            test_data_df.loc[test_data_df["Total Evaluation_Emotion"] == 'neutral', "Total Evaluation_Emotion"] = 4
            test_data_df.loc[test_data_df["Total Evaluation_Emotion"] == 'sad', "Total Evaluation_Emotion"] = 5
            test_data_df.loc[test_data_df["Total Evaluation_Emotion"] == 'surprise', "Total Evaluation_Emotion"] = 6
            test_data_df = test_data_df.astype({"Total Evaluation_Emotion":'int'})
            
            file_names = test_data_df['Segment ID'].values


            file_name_pst = []
            file_path_list = []

            for file in file_names:
                seg_file = file.split('_')
                file_name_pst.append(file)
                file_path_list.append(parent_path20+seg_file[0][-2:]+'/'+file+'.wav')
            
            test_speech_arrays = []
            for audio_file in file_path_list:
                speech_array,_ = torchaudio.load(audio_file)
                test_speech_arrays.extend(speech_array)

            test_labels = test_data_df["Total Evaluation_Emotion"].values


            test_acoustic = test_speech_arrays

            # a sentinel epsilon for safe division, without it we will replace illegal values with a constant
            EPS = 1e-6

            # place holders for the final train/test dataset
            self.train = train = []
            self.test = test = []

            num_drop = 0 # a counter to count how many data points went into some processing issues

            # Iterate over all possible utterances
            for key in range(len(train_labels)):

                label = train_labels[key]
                _acoustic = train_acoustic[key]


                # remove nan values
                label = np.array([np.nan_to_num(label)])[:, np.newaxis]
                _acoustic = np.nan_to_num(_acoustic)
                
                acoustic = np.asarray(_acoustic)

                # z-normalization per instance and remove nan/infs
                acoustic = np.nan_to_num((acoustic - acoustic.mean(0, keepdims=True)) / (EPS + np.std(acoustic, axis=0, keepdims=True)))

                train.append((acoustic, label))

            to_pickle(train, DATA_PATH + '/train_wav_20_fintuning.pkl')
           
            for key in range(len(test_labels)):

                label = test_labels[key]
                _acoustic = test_acoustic[key]

                # remove nan values
                label = np.array([np.nan_to_num(label)])[:, np.newaxis]
                _acoustic = np.nan_to_num(_acoustic)

                acoustic = np.asarray(_acoustic)

                # z-normalization per instance and remove nan/infs
                acoustic = np.nan_to_num((acoustic - acoustic.mean(0, keepdims=True)) / (EPS + np.std(acoustic, axis=0, keepdims=True)))                
                
                test.append((acoustic, label))
            
            to_pickle(test, DATA_PATH + '/test_wav_20_fintuning.pkl')

            print(f"Total number of {num_drop} datapoints have been dropped.")
 


    def get_data(self, mode):

        if mode == "train":
            return self.train
        elif mode == "test":
            return self.test
        
        else:
            print("Mode is not set properly (train/test)")
            exit()