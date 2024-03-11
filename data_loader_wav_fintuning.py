import random
import numpy as np
from tqdm import tqdm_notebook
from collections import defaultdict

import torch
import torch.nn as nn 

from torch.utils.data import DataLoader, Dataset
from transformers import Wav2Vec2Processor

from create_dataset_wav_fintuning import KemDY20


wav_processor = Wav2Vec2Processor.from_pretrained("kresnik/wav2vec2-large-xlsr-korean")


class MSADataset(Dataset): 
    def __init__(self, config):

        ## Fetch dataset
        if "kemdy19" in str(config.data_dir).lower():
            dataset = KemDY20(config)
        elif "kemdy20" in str(config.data_dir).lower():
            dataset = KemDY20(config)
        else:
            print("Dataset not defined correctly")
            exit()
        
        self.data = dataset.get_data(config.mode)
        self.len = len(self.data)

        if config.mode == 'train':
          config.acoustic_size = 563472
        elif config.mode =='test':
        #563472
          config.acoustic_size = 563472
  
          

        

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len



def get_loader_fine(config, shuffle=True):
    """Load DataLoader of given DialogDataset"""

    dataset = MSADataset(config)
    
    print(config.mode)
    config.data_len = len(dataset)


    def collate_fn(batch):
        '''
        Collate functions assume batch = [Dataset[i] for i in index_set]
        '''
        
        # get the data out of the batch - use pad sequence util functions from PyTorch to pad things
        labels = torch.cat([torch.from_numpy(sample[1]) for sample in batch], dim=0)
        #acoustic = pad_sequence([torch.FloatTensor(sample[0]) for sample in batch])
        
        wav2_input = []
        wav2_att_mask = []
        wav_features_list = []
        for sample in batch:
            wav_features = wav_processor(sample[0], sampling_rate=16000, return_tensors="pt", padding=True)
            wav_features_list.append(wav_features)

        input_features = [{'input_values': wav_features['input_values'][0]} for wav_features in wav_features_list]
        wav_batch = wav_processor.pad(input_features, padding=True, max_length = None, pad_to_multiple_of=None, return_tensors='pt')


        wav2_input = wav_batch['input_values']
        wav2_att_mask = wav_batch['attention_mask']
  

        return  wav2_input, wav2_att_mask, labels


    data_loader = DataLoader(
        dataset=dataset,
        batch_size=4,
        shuffle=shuffle,
        collate_fn=collate_fn)

    return data_loader
