import argparse
import os
import random
import time
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import DataLoader, DataListLoader,Batch
from torch_geometric.nn import DataParallel
from sklearn import metrics
import torch.nn.functional as F
from torch.optim import Adam
import torch.nn as nn
import torch_geometric
from dataset import *
from utils import *
from model import *
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from preprocess_dataset import *
from collections import OrderedDict


def eval(args, model, device, loader):
    model = model.to(device)
    model.eval()
    y_pre = []
    with torch.no_grad():
        for step, batch in enumerate(tqdm(loader, desc="Iteration")):
            batch = batch.to(device)
            target_len = batch.pr_len
            y_idx = torch.zeros(target_len[0]).long()
            for i, e in enumerate(target_len):
                if i > 0:
                    y_idx = torch.cat([y_idx, torch.full((e.item(),), i).long()])
            y_idx = y_idx.to(device)
            batch.protein = torch_geometric.utils.to_dense_batch(batch.protein, y_idx)[0]
            pred = model(batch)
            pred = torch.argmax(pred,dim=1)
            y_pre.extend(pred.cpu().squeeze().numpy())
    return np.array(y_pre)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default='dataset/', help="all data dir")
    parser.add_argument("--dataset", type=str, default='drugbank')
    parser.add_argument('--seed', default=33, type=int)
    parser.add_argument("--metric", type=str, default='roc')

    parser.add_argument("--hid", type=int, default=32, help="hidden size of transformer model")
    parser.add_argument('--heads', default=4, type=int)
    parser.add_argument('--depth', default=1, type=int)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument('--protein', type=str, default='cdk46', help='cdk12,cdk46')
    parser.add_argument('--target',type=str,default='IC50',help='Kd,Ki')
    parser.add_argument('--num_iters',default=3,type=int)

    parser.add_argument("--batch_size", type=int, default=32, help="number of batch_size")
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--num_workers", type=int, default=0, help="dataloader worker size")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate of adam")
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--clip_norm", type=float, default=0.0)
    parser.add_argument('--lr_scheduler_patience', default=20, type=int)
    parser.add_argument('--early_stop_patience', default=-1, type=int)
    parser.add_argument('--lr_decay', default=0.98, type=float)
    parser.add_argument('--focalloss', default=False, action="store_true")
    parser.add_argument('--single_task_mseloss', default=False)

    parser.add_argument('--load', default=None, type=str)
    parser.add_argument('--eval', default=False, action="store_true")
    parser.add_argument("--exps_dir", default='test',type=str, help="out/")
    parser.add_argument('--exp_name', default=None, type=str)
    parser.add_argument('--test_path', default=None, type=str)
    parser.add_argument('--gnn', default="pretrain_model/ckpt_8393.pt", type=str)
    parser.add_argument("--local_rank",
                        type=int,
                        default=os.getenv('LOCAL_RANK', -1),
                        help="local_rank for distributed training on gpus")
    args = parser.parse_args()

    device = torch.device("cuda:0")
    seed_set(seed=args.seed)
    # if args.exp_name is None:
    #     args.exp_name = time.strftime("%m-%d-%H-%M")
    #
    # args.exp_path = os.path.join(args.exps_dir, args.exp_name)
    # if not os.path.exists(args.exp_path):
    #     os.makedirs(args.exp_path)
    #     print('makedir exp path')
    # else:
    #     print('exist exp path')
    print('Preparing dataset...')
    df_target = pd.read_csv('dataset/protein.csv')
    protein = df_target[df_target['name']==args.protein]['sequence'].values
    assert len(protein)==1
    protein = protein[0]

    pyg_dataset = TestDataset('dataset/'+args.dataset, args.dataset, target=protein, target_name=args.protein)
    loader = DataLoader(pyg_dataset,batch_size=args.batch_size, shuffle=False, num_workers=4)
    df_drug = pd.read_csv(os.path.join('dataset/'+args.dataset,'processed/valid_smile.csv'))

    print('Preparing model...')
    n_word = pyg_dataset.n_word
    for target in ['human','celegans']:
        model = CompoundProteinInteractionPrediction(n_word)
        seed=33

        state_dict = torch.load('test/{}/seed_{}/best_model_{}.ckpt'.format(target,seed,seed),map_location=torch.device('cpu'))
        # state_dict_new = OrderedDict()
        # for k,v in state_dict.items():
        #     name = k[7:]
        #     state_dict_new[name]=v

        model.load_state_dict(state_dict['model_state_dict'])
        model = model.to(device)
        y_pre = eval(args, model, device, loader)
        # y_pre = convert_y_unit(y_pre, from_='p', to_='nM')
        df_drug[target]=y_pre
    df_drug.to_csv('./prediction_{}.csv'.format(args.protein))


if __name__ == "__main__":
    main()








