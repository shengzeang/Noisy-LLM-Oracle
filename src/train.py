import torch
from ogb.nodeproppred import Evaluator
from torch.optim.lr_scheduler import  _LRScheduler
from torch_geometric.utils import index_to_mask, subgraph
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from src.model import str2model
from src.utils import seed_everything



def train_pipeline(args ,mask ,dataset ,pre_labels ):
    print("#---------------------------mode_training------------------------------#")
    device="cpu" if args.device<0 else "cuda:"+str(args.device)
    if args.dataset == 'ogbn-arxiv':
        features=dataset.x.to(device)
        labels=dataset.y.squeeze(1).to(device)
    else:
        features=dataset.feature.to(device)
        labels=dataset.labels.to(device)
    pre_labels=pre_labels.to(device)
    edge_index=dataset.edge_index.to(device)
    train_mask,val_mask,test_mask = mask

    # print(features.shape)
    
    this_test_acc=[]
    this_train_acc=[]
    this_best_test_acc=[]
    for seed in range(args.seeds):
        seed_everything(seed)
        model = str2model[args.model](args).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay=args.weight_decay)
        loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)   
        
        best_test_acc=0            
        for i in tqdm(range(args.epochs)):
            train_loss = train(model, features,edge_index,pre_labels, optimizer, loss_fn, train_mask)
            test_acc = test(model, features,edge_index, labels, test_mask)
            if test_acc>best_test_acc:
                best_test_acc=test_acc
                
        train_acc = test(model, features,edge_index, pre_labels, train_mask)
        test_acc = test(model, features,edge_index, labels, test_mask)
        this_test_acc.append(test_acc)
        this_train_acc.append(train_acc)
        this_best_test_acc.append(best_test_acc)

    return this_train_acc,this_test_acc,this_best_test_acc

def train(model, features,edge_index, pre_labels,optimizer, loss_fn, train_mask):
    preds = model(features,edge_index)
    train_loss = loss_fn(preds[train_mask], pre_labels[train_mask])
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()
    return train_loss

@torch.no_grad()
def test(model, features,edge_index,labels,test_mask):
    model.eval()
    out = model(features,edge_index)
    y_pred = out.argmax(dim=-1, keepdim=True)
    evaluator = Evaluator(name='ogbn-arxiv')
    acc = evaluator.eval({
        'y_true': labels[test_mask].unsqueeze(-1),
        'y_pred': y_pred[test_mask],
    })['acc']
    return acc



class WarmupExpLR(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, total_epochs, gamma=0.1, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.gamma = gamma
        super(WarmupExpLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            return [group['lr'] for group in self.optimizer.param_groups]
        else:
            return [group['lr'] * self.gamma
                for group in self.optimizer.param_groups]
    
    def _get_closed_form_lr(self):
        return [base_lr * self.gamma ** self.last_epoch
                for base_lr in self.base_lrs]

def get_optimizer(args, model):
    if args.model_name == 'LP':
        return None, None
    if args.optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay=args.weight_decay)
        scheduler = None 
    elif args.optim == 'radam':
        optimizer = torch.optim.RAdam(model.parameters(), lr = args.lr, weight_decay=args.weight_decay)
        scheduler = WarmupExpLR(optimizer, args.warmup, total_epochs=args.epochs, gamma=args.lr_gamma)
    return optimizer, scheduler



















































































































































