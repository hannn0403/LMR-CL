import random
import numpy as np
from tqdm import tqdm_notebook
from collections import defaultdict 
 
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from create_dataset_only_bert_feature_20 import KemDY20


koEletra_tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")


class MSADataset(Dataset):
    def __init__(self, config):

        ## Dataset
        if "kemdy20" in str(config.data_dir).lower():
            dataset = KemDY20(config)
        else:
            print("Dataset not defined correctly")
            exit()
        
        self.data = dataset.get_data(config.mode)
        self.len = len(self.data)
        self.y = np.array([sample[1].squeeze(0).squeeze(0) for sample in self.data])

 
        config.phy_size = 19
        if config.mode == 'train':
          config.acoustic_size = 563472
        elif config.mode =='test':
        #563472
          config.acoustic_size = 563472

         

        

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len



def get_loader(config, shuffle=True):
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
        
        sentences = []
        phy = torch.FloatTensor(np.array([sample[0][1] for sample in batch]))
        acoustic = torch.FloatTensor(np.array([sample[0][2] for sample in batch]))
    
        bert_details = []

        for sample in batch:

            text = [str(sample[0][0])]
            encoded_bert_sent = koEletra_tokenizer(
                text, max_length=256, return_tensors='pt', truncation=True, add_special_tokens=True, pad_to_max_length=True)
            bert_details.append(encoded_bert_sent)
            

        # Bert things are batch_first
   
        bert_sentences = torch.LongTensor([sample["input_ids"][0].numpy() for sample in bert_details])
        bert_sentence_att_mask = torch.LongTensor([sample["attention_mask"][0].numpy() for sample in bert_details])
        

        # lengths are useful later in using RNNs
        lengths = torch.LongTensor([1 for sample in batch])

        return sentences, phy, acoustic, labels, lengths, bert_sentences, bert_sentence_att_mask

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn)


    return data_loader
