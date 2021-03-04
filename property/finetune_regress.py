import argparse
from loader import MoleculeDataset,DataLoaderMasking
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from model import MolGT_graphpred
from sklearn.metrics import roc_auc_score
from splitters import scaffold_split,random_scaffold_split,random_split,scaffold_split_fp
import pandas as pd
import os
from util import *
import warnings,random
warnings.filterwarnings("ignore")

def disable_rdkit_logging():
    """
    Disables RDKit whiny logging.
    """
    import rdkit.rdBase as rkrb
    import rdkit.RDLogger as rkl
    logger = rkl.logger()
    logger.setLevel(rkl.ERROR)
    rkrb.DisableLog('rdApp.error')

disable_rdkit_logging()
criterion = torch.nn.MSELoss()

def train(args, model, device, loader, optimizer):
    model = model.to(device)
    model.train()
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        pred = model(batch)
        y = batch.y.view(pred.shape).to(torch.float64)
        loss = criterion(pred.double(), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def eval(args, model, device, loader):
    model.eval()
    y_loss = []
    y_rmse = []
    with torch.no_grad():
        for step, batch in enumerate(tqdm(loader, desc="Iteration")):
            batch = batch.to(device)
            pred = model(batch)
            y = batch.y.view(pred.shape).to(torch.float64)
            loss = criterion(pred.double(), y)
            y_loss.append(loss.item())
            y_rmse.append(torch.sqrt(loss).item())
    return np.mean(y_loss),np.mean(y_rmse)



def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--lr_decay', type=float, default=0.995,
                        help='learning rate decay (default: 0.995)')
    parser.add_argument('--lr_scale', type=float, default=1,
                        help='relative learning rate for the feature extraction layer (default: 1)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=768,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--graph_pooling', type=str, default="mean",
                        help='graph level pooling (sum, mean, max, set2set, attention)')
    parser.add_argument('--heads', type=int, default=12,
                        help='multi heads (default: 4)')
    parser.add_argument('--num_message_passing', type=int, default=3,
                        help='message passing steps (default: 3)')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--dataset', type=str, default = 'tox21', help='root directory of dataset. For now, only classification.')
    parser.add_argument('--input_model_file', type=str, default='pretrained_model/MolGNet.pt',
                        help='filename to read the model (if there is any)')
    parser.add_argument('--filename', type=str, default = '', help='output filename')
    parser.add_argument('--seed', type=int, default=177, help = "Seed for splitting the dataset.")
    parser.add_argument('--runseed', type=int, default=0, help = "Seed for minibatch selection, random initialization.")
    parser.add_argument('--split', type = str, default="scaffold", help = "random or scaffold or random_scaffold")
    parser.add_argument('--eval_train', type=int, default = 0, help='evaluating training or not')
    parser.add_argument('--num_workers', type=int, default = 4, help='number of workers for dataset loading')
    parser.add_argument('--iters', type=int, default=10, help='number of run seeds')
    parser.add_argument('--processed_file', type=str, default=None)
    parser.add_argument('--raw_file', type=str, default=None)
    parser.add_argument('--cpu', default=False, action="store_true")
    parser.add_argument('--exp', type=str, default='', help='output filename')
    parser.add_argument('--data_dir', type=str, default="")
    args = parser.parse_args()

    device = torch.device("cuda:0") if torch.cuda.is_available() and not args.cpu else torch.device("cpu")

    if args.dataset == "freesolv":
        # args.seed =219
        # args.runseed = 142
        args.batch_size = 32
        args.lr = 0.0001
        args.lr_decay = 0.99
        args.dropout_ratio = 0
        args.graph_pooling = 'mean'
        args.data_dir = 'data/downstream/'

    elif args.dataset == "esol":
        args.batch_size = 32
        args.lr = 0.001
        args.lr_decay = 0.995
        args.dropout_ratio = 0.5
        args.graph_pooling = 'set2set'
        args.data_dir = 'data/downstream/'

    elif args.dataset == "lipophilicity":
        args.batch_size = 32
        args.lr = 0.0001
        args.lr_decay = 0.99
        args.dropout_ratio = 0
        args.graph_pooling = 'set2set'

    for i in range(args.iters):
        seed=args.seed+i
        runseed=args.runseed
        torch.manual_seed(runseed)
        np.random.seed(runseed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(runseed)
        #Bunch of classification tasks
        num_tasks = 1
        transform = Compose(
            [
                Self_loop(), Add_seg_id(), Add_collection_node(num_atom_type=119, bidirection=False)
            ]
        )
        dataset = MoleculeDataset(args.data_dir  + args.dataset, dataset=args.dataset, transform=transform
                                  )


        smiles_list = pd.read_csv(args.data_dir  + args.dataset + '/processed/smiles.csv')['smiles'].tolist()
        train_dataset, valid_dataset, test_dataset = random_scaffold_split(dataset, smiles_list, null_value=0,
                                                                    frac_train=0.8, frac_valid=0.1, frac_test=0.1,
                                                                           seed=seed)


        train_loader = DataLoaderMasking(train_dataset, batch_size=args.batch_size, shuffle=True,
                                         num_workers=args.num_workers)
        val_loader = DataLoaderMasking(valid_dataset, batch_size=args.batch_size, shuffle=False,
                                       num_workers=args.num_workers)
        test_loader = DataLoaderMasking(test_dataset, batch_size=args.batch_size, shuffle=False,
                                        num_workers=args.num_workers)

        # set up model
        model = MolGT_graphpred(args.num_layer, args.emb_dim, args.heads, args.num_message_passing, num_tasks,
                                drop_ratio=args.dropout_ratio, graph_pooling=args.graph_pooling)
        if not args.input_model_file == "":
            model.from_pretrained(args.input_model_file)
            print('Pretrained model loaded')

        model.to(device)

        # set up optimizer
        # different learning rate for different part of GNN
        model_param_group = []
        model_param_group.append({"params": model.gnn.parameters()})
        if args.graph_pooling == "attention":
            model_param_group.append({"params": model.pool.parameters(), "lr": args.lr * args.lr_scale})
        model_param_group.append({"params": model.graph_pred_linear.parameters(), "lr": args.lr * args.lr_scale})
        optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)
        print(optimizer)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay)

        train_acc_list = []
        val_acc_list = []
        test_acc_list = []

        exp_path = '{}/{}_seed{}/'.format(args.exp,args.dataset, seed)
        if not os.path.exists(exp_path):
            os.makedirs(exp_path)
        best_rmse=float('inf')
        for epoch in range(1, args.epochs+1):
            print("====epoch " + str(epoch))
            train(args, model, device, train_loader, optimizer)
            scheduler.step()
            print("====Evaluation")
            train_loss,train_rmse = eval(args, model, device, train_loader)
            val_loss,val_acc = eval(args, model, device, val_loader)
            test_loss,test_acc = eval(args, model, device, test_loader)
            print("RMSE: train: %f val: %f test: %f" %(train_loss, val_acc, test_acc))
            if val_acc<=best_rmse:
                best_rmse=val_acc
                torch.save(model.state_dict(), exp_path + "model_seed{}.pkl".format(args.seed))
                print('saved')

            val_acc_list.append(val_acc)
            test_acc_list.append(test_acc)
            train_acc_list.append(train_rmse)

        df = pd.DataFrame({'train':train_acc_list,'valid':val_acc_list,'test':test_acc_list})
        df.to_csv(exp_path+'{}_seed{}.csv'.format(args.dataset,seed))

        best_epoch = np.argmax(val_acc_list)
        test_acc_at_best_val = test_acc_list[best_epoch]
        print("The test auc at best valid (epoch {}) is {} at seed {}".format(best_epoch,test_acc_at_best_val,args.runseed))


if __name__ == "__main__":
    main()
