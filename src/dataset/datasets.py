import torch 
import os.path as osp
import numpy as np 
from tqdm import tqdm
from src.utils import *

class BaseDataset:
    def __init__(self, args):
        self.dataset = args.dataset

        if self.dataset == 'citeseer':
            self.data = torch.load(osp.join("data/", "citeseer.pt"), map_location='cpu')
        elif self.dataset == 'cora':
            self.data = torch.load(osp.join("data/", "cora.pt"), map_location='cpu')
        elif self.dataset == 'pubmed':
            self.data = torch.load(osp.join("data/", "pubmed.pt"), map_location='cpu')
        elif self.dataset == 'wikics':
            self.data = torch.load(osp.join("data/", "wikics.pt"), map_location='cpu')
        else:
            pass

        full_mapping = load_mapping()
        self.label_names = [full_mapping[self.dataset][x] for x in self.data.label_names]
        self.label_names = [x.lower() for x in self.label_names]
        self.feature=self.data.x
        self.labels =self.data.y
        self.edge_index=self.data.edge_index
        self.num_class= self.data.y.max().item() + 1
        self.pre_labels=None
        self.labels_sim=None
        
    def get_rawtext(self):
        
        return self.data.raw_texts
    
    def __len__(self):
        return len(self.data.x)
    
    def __getitem__(self, index):
        return self.data.raw_texts[index]




























