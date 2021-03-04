import argparse,os
from loader import MoleculeDataset,DataLoaderMasking
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score,accuracy_score,precision_score,recall_score,precision_recall_curve,f1_score
from sklearn.preprocessing import label_binarize
from tqdm import tqdm
import numpy as np
from model import MolGT_graphpred
from sklearn.metrics import roc_auc_score
from splitters import scaffold_split, random_scaffold_split, random_split, scaffold_split_fp,cv_random_split
from util import *
import warnings, random
from sklearn import metrics
warnings.filterwarnings("ignore")


# Workaround because python functions are not picklable
class WorkerInitObj(object):
    def __init__(self, seed):
        self.seed = seed

    def __call__(self, id):
        np.random.seed(seed=self.seed + id)
        random.seed(self.seed + id)


def cal_acc(pred,y_true):
    acc = accuracy_score(y_true,pred) #acc
    f1 = f1_score(y_true,pred,average='macro') #macro-F1
    prec = precision_score(y_true,pred,average='macro') #macro-Precision
    rec = recall_score(y_true,pred,average='macro') #macro-Recall
    return acc,f1,prec,rec


def eval_twosides(model, device, loader):
    model.eval()
    y_true = []
    pred_score = []
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        with torch.no_grad():
            pred = model(batch)
        y=batch.y.view(pred.shape).to(torch.long).cpu().numpy()
        pred=torch.sigmoid(pred).cpu().numpy()
        y_true.append(y)
        pred_score.append(pred)
    y_true = np.concatenate(y_true,axis=0)
    y_test_one_hot = label_binarize(y_true,np.arange(86))
    pred_score = np.concatenate(pred_score,axis=0)
    pred_type = np.int64(pred_score>0.5)
    result = cal_acc(pred_type,y_true)
    return result

def cal_metrics(y_prob,y_true):
    auc = roc_auc_score(y_true,y_prob)
    prc = metrics.auc(metrics.precision_recall_curve(y_true,y_prob)[1],
                    metrics.precision_recall_curve(y_true,y_prob)[0])
    pred= np.int64(y_prob > 0.5)
    f1 = f1_score(y_true, pred, average='macro')
    return [auc,prc,f1]

def eval_biosnap(model, device, loader):
    model.eval()
    y_true = []
    pred_score = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        with torch.no_grad():
            pred = model(batch)
        y=batch.y.view(pred.shape).to(torch.long).cpu().numpy()
        pred=torch.sigmoid(pred).cpu().numpy()
        y_true.append(y)
        pred_score.append(pred)
    y_true = np.concatenate(y_true,axis=0)
    pred_score = np.concatenate(pred_score,axis=0)
    result = cal_metrics(pred_score,y_true)
    return result

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--lr_decay', type=float, default=0.995,
                        help='learning rate decay (default: 0.995)')
    parser.add_argument('--lr_scale', type=float, default=1,
                        help='relative learning rate for the feature extraction layer (default: 1)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--loss_type', type=str, default="bce")
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=768,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--heads', type=int, default=12,
                        help='multi heads (default: 4)')
    parser.add_argument('--num_message_passing', type=int, default=3,
                        help='message passing steps (default: 3)')
    parser.add_argument('--dropout_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--graph_pooling', type=str, default="collection",
                        help='graph level pooling (collection,sum, mean, max, set2set, attention)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--dataset', type=str, default='twosides',
                        help='root directory of dataset. For now, only classification.')
    parser.add_argument('--model_dir', type=str, default='pretrained_model/MolGNet.pt',
                        help='filename to read the model (if there is any)')
    parser.add_argument('--filename', type=str, default='', help='output filename')
    parser.add_argument('--seed', type=int, default=42, help="Seed for splitting the dataset.")
    parser.add_argument('--runseed', type=int, default=0, help="Seed for minibatch selection, random initialization.")
    parser.add_argument('--split', type=str, default="scaffold", help="random or scaffold or random_scaffold")
    parser.add_argument('--eval_train', type=int, default=0, help='evaluating training or not')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers for dataset loading')
    parser.add_argument('--iters', type=int, default=1, help='number of run seeds')
    parser.add_argument('--log_file', type=str, default=None)
    parser.add_argument('--log_freq', type=int, default=0)
    parser.add_argument('--KFold', type=int, default=5, help='number of folds for cross validation')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% "
                             "of training.")
    parser.add_argument('--cpu', default=False, action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda:0") if torch.cuda.is_available() and not args.cpu else torch.device("cpu")
    args.seed = args.seed
    args.runseed = args.runseed
    torch.manual_seed(args.runseed)
    np.random.seed(args.runseed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.runseed)

    args.model_dir = os.path.join(args.model_dir,args.dataset)

    num_tasks = 1
    dataset = MoleculeDataset("data/downstream/"  + args.dataset, dataset=args.dataset, transform=None)
    if args.dataset=='twosides':
        print('Run 5-fold cross validation')
        for fold  in range(args.KFold):
            train_dataset, test_dataset = cv_random_split(dataset, fold,5)
            test_loader = DataLoaderMasking(test_dataset, batch_size=args.batch_size, shuffle=False,
                                            num_workers=args.num_workers)
            # set up model
            model = MolGT_graphpred(args.num_layer, args.emb_dim, args.heads, args.num_message_passing, num_tasks,
                                    drop_ratio=args.dropout_ratio, graph_pooling=args.graph_pooling)

            model.load_state_dict(torch.load(os.path.join(args.model_dir,'Fold_{}.pkl'.format(fold))))
            model.to(device)

            acc,f1,prec,rec = eval_twosides(model, device, test_loader)
            print('Dataset:{}, Fold:{}, precision:{}, recall:{}, F1:{}'.format(args.dataset,fold,prec,rec,f1))

    if args.dataset=='biosnap':
        print('Run three random split')
        for i in range(3):
            seed =args.seed+i
            train_dataset, valid_dataset, test_dataset = random_split(dataset, null_value=0, frac_train=0.7, frac_valid=0.1,
                                                                      frac_test=0.2, seed=seed)
            test_loader = DataLoaderMasking(test_dataset, batch_size=args.batch_size, shuffle=False,
                                            num_workers=args.num_workers)
            model = MolGT_graphpred(args.num_layer, args.emb_dim, args.heads, args.num_message_passing, num_tasks,
                                    drop_ratio=args.dropout_ratio, graph_pooling=args.graph_pooling)
            model.load_state_dict(torch.load(os.path.join(args.model_dir,'model_seed{}.pkl'.format(seed))))
            model.to(device)
            auc,prc,f1 = eval_biosnap(model, device, test_loader)
            print('Dataset:{}, Fold:{}, AUC:{}, PRC:{}, F1:{}'.format(args.dataset,i,auc,prc,f1))

if __name__ == "__main__":
    main()
