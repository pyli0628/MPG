from tqdm import tqdm
import argparse
import numpy as np
from model import MolGT_graphpred
from sklearn.metrics import roc_auc_score
import os,shutil,glob
from loader import *
from util import *
import torch

def eval(model, device, loader):
    model.eval()
    y_true = []
    y_scores = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        with torch.no_grad():
            pred = model(batch)
        y_true.append(batch.y.view(pred.shape))
        y_scores.append(pred)

    y_true = torch.cat(y_true, dim = 0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim = 0).cpu().numpy()

    roc_list = []
    for i in range(y_true.shape[1]):
        #AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == -1) > 0:
            is_valid = y_true[:,i]**2 > 0
            roc_list.append(roc_auc_score((y_true[is_valid,i] + 1)/2, y_scores[is_valid,i]))

    if len(roc_list) < y_true.shape[1]:
        print("Some target is missing!")
        print("Missing ratio: %f" %(1 - float(len(roc_list))/y_true.shape[1]))

    return sum(roc_list)/len(roc_list) #y_true.shape[1]

def eval_regress(model, device, loader):
    criterion = torch.nn.MSELoss()
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
    return np.mean(y_rmse)

parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
parser.add_argument('--dataset', type=str, default = 'tox21')
parser.add_argument('--model_dir', type=str, default = 'tox21')
parser.add_argument('--cpu', default=False, action="store_true")
args = parser.parse_args()

dataset = args.dataset
args.model_dir = os.path.join(args.model_dir,dataset)
device = torch.device("cuda:0") if torch.cuda.is_available() and not args.cpu else torch.device("cpu")


if dataset == "tox21":
    num_tasks = 12
    graph_pooling = 'collection'
    dropout_ratio = 0.2
elif dataset == "bace":
    num_tasks = 1
    graph_pooling = 'collection'
    dropout_ratio = 0
elif dataset == "bbbp":
    num_tasks = 1
    graph_pooling = 'attention'
    dropout_ratio = 0.2
elif dataset == "toxcast":
    num_tasks = 617
    graph_pooling = 'collection'
    dropout_ratio = 0.2
elif dataset == "sider":
    num_tasks = 27
    graph_pooling = 'collection'
    dropout_ratio = 0.2
elif dataset == "clintox":
    num_tasks = 2
    graph_pooling='set2set'
    dropout_ratio = 0.2
elif dataset == "freesolv":
    num_tasks = 1
    graph_pooling='mean'
    dropout_ratio = 0
elif dataset == "esol":
    num_tasks = 1
    graph_pooling='set2set'
    dropout_ratio = 0.5
elif dataset == "lipophilicity":
    num_tasks = 1
    graph_pooling='set2set'
    dropout_ratio = 0
else:
    raise ValueError("Invalid dataset name.")

transform = Compose(
    [
        Self_loop(), Add_seg_id(), Add_collection_node(num_atom_type=119, bidirection=False)
    ]
)

models = glob.glob(args.model_dir+'/*/*pkl')

for fold in range(3):
    test_file= glob.glob(os.path.dirname(models[fold])+'/test.csv')[0]
    test_dataset = TestDataset(file_path=test_file, dataset=dataset, transform=transform)
    test_loader = DataLoaderMasking(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    # set up model
    model = MolGT_graphpred(5, 768, 12, 3, num_tasks,
                            drop_ratio=dropout_ratio, graph_pooling=graph_pooling)


    model.load_state_dict(torch.load(models[fold]))

    model.to(device)
    if dataset in ['freesolv', 'esol', 'lipophilicity']:
        test_rmse = eval_regress(model, device=device, loader=test_loader)
        print('rmse', test_rmse)
    else:
        test_acc = eval(model, device=device, loader=test_loader)
        print('acc', test_acc)