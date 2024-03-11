import sys
import os
import re
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook
from collections import defaultdict
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


        # If cached data if already exists
        try:
            self.train = load_pickle(DATA_PATH + '/train_20.pkl')
            self.test = load_pickle(DATA_PATH + '/test_20.pkl')
 

        except:


            # create folders for storing the data 
            if not os.path.exists(DATA_PATH):
                check_call(' '.join(['mkdir', '-p', DATA_PATH]), shell=True)
 
            data_df = pd.read_csv('./Data/text_train.csv', index_col=0)

            data_df.loc[data_df["Total Evaluation_Emotion"] == 'angry', "Total Evaluation_Emotion"] = 0
            data_df.loc[data_df["Total Evaluation_Emotion"] == 'disgust', "Total Evaluation_Emotion"] = 1
            data_df.loc[data_df["Total Evaluation_Emotion"] == 'disqust', "Total Evaluation_Emotion"] = 1
            data_df.loc[data_df["Total Evaluation_Emotion"] == 'fear', "Total Evaluation_Emotion"] = 2
            data_df.loc[data_df["Total Evaluation_Emotion"] == 'happy', "Total Evaluation_Emotion"] = 3
            data_df.loc[data_df["Total Evaluation_Emotion"] == 'neutral', "Total Evaluation_Emotion"] = 4
            data_df.loc[data_df["Total Evaluation_Emotion"] == 'sad', "Total Evaluation_Emotion"] = 5
            data_df.loc[data_df["Total Evaluation_Emotion"] == 'surprise', "Total Evaluation_Emotion"] = 6
            data_df =data_df.astype({"Total Evaluation_Emotion":'int'})
            
            
            phy_df = pd.read_csv('./Data/Biosignal_train.csv', index_col=0)

            

            train_labels = data_df["Total Evaluation_Emotion"].values 
            train_word_id = data_df['content'].values
            train_acoustic = np.load('./Data/wav_feature_20_fintuning_train.npy').squeeze(1)
            
            train_phy = phy_df.iloc[:,1:-1].values


            # a sentinel epsilon for safe division, without it we will replace illegal values with a constant

            test_data_df = pd.read_csv('./Data/text_test.csv', index_col=0)

            test_data_df.loc[test_data_df["Total Evaluation_Emotion"] == 'angry', "Total Evaluation_Emotion"] = 0
            test_data_df.loc[test_data_df["Total Evaluation_Emotion"] == 'disgust', "Total Evaluation_Emotion"] = 1
            test_data_df.loc[test_data_df["Total Evaluation_Emotion"] == 'disqust', "Total Evaluation_Emotion"] = 1
            test_data_df.loc[test_data_df["Total Evaluation_Emotion"] == 'fear', "Total Evaluation_Emotion"] = 2
            test_data_df.loc[test_data_df["Total Evaluation_Emotion"] == 'happy', "Total Evaluation_Emotion"] = 3
            test_data_df.loc[test_data_df["Total Evaluation_Emotion"] == 'neutral', "Total Evaluation_Emotion"] = 4
            test_data_df.loc[test_data_df["Total Evaluation_Emotion"] == 'sad', "Total Evaluation_Emotion"] = 5
            test_data_df.loc[test_data_df["Total Evaluation_Emotion"] == 'surprise', "Total Evaluation_Emotion"] = 6
            test_data_df = test_data_df.astype({"Total Evaluation_Emotion":'int'})
            
            
            test_phy_df = pd.read_csv('./Data/Biosignal_test.csv', index_col=0)

            test_labels = test_data_df["Total Evaluation_Emotion"].values 
            test_word_id = test_data_df['content'].values
            test_acoustic = np.load('./Data/wav_feature_20_fintuning_test.npy').squeeze(1)
            test_phy = test_phy_df.iloc[:,1:-1].values



            EPS = 1e-6

            # place holders for the final train/test dataset
            self.train = train = []

            self.test = test = []


            num_drop = 0 # a counter to count how many data points went into some processing issues


            for key in range(len(train_labels)):

                label = train_labels[key]
                _word_id = str(train_word_id[key])
                _acoustic = train_acoustic[key]
                _phy = train_phy[key]

                # remove nan values
                label = np.array([np.nan_to_num(label)])[:, np.newaxis]
                _phy = np.nan_to_num(_phy)
                _acoustic = np.nan_to_num(_acoustic)
                
                words = _word_id
                phy = np.asarray(_phy)
                acoustic = np.asarray(_acoustic)

                # z-normalization per instance and remove nan/infs
                phy = np.nan_to_num((phy - phy.mean(0, keepdims=True)) / (EPS + np.std(phy, axis=0, keepdims=True)))
     
                train.append(((words, phy, acoustic), label))
 
            
            for key in range(len(test_labels)):

                label = test_labels[key]
                _word_id = str(test_word_id[key])
                _acoustic = test_acoustic[key]
                _phy = test_phy[key]

                # remove nan values
                label = np.array([np.nan_to_num(label)])[:, np.newaxis]
                _phy = np.nan_to_num(_phy)
                _acoustic = np.nan_to_num(_acoustic)

                words = _word_id
                phy = np.asarray(_phy)
                acoustic = np.asarray(_acoustic)

                # z-normalization per instance and remove nan/infs
                phy = np.nan_to_num((phy - phy.mean(0, keepdims=True)) / (EPS + np.std(phy, axis=0, keepdims=True)))
         
                
                test.append(((words, phy, acoustic), label))


            print(f"Total number of {num_drop} datapoints have been dropped.")
 

            # Save pickles
            to_pickle(train, DATA_PATH + '/train_20.pkl')
            to_pickle(test, DATA_PATH + '/test_20.pkl')
            
      

    def get_data(self, mode):

        if mode == "train":
            return self.train
        elif mode == "test":
            return self.test
        
        else:
            print("Mode is not set properly (train/test)")
            exit()