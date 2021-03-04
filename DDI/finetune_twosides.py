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
from optimization import BertAdam, warmup_linear

warnings.filterwarnings("ignore")


# Workaround because python functions are not picklable
class WorkerInitObj(object):
    def __init__(self, seed):
        self.seed = seed

    def __call__(self, id):
        np.random.seed(seed=self.seed + id)
        random.seed(self.seed + id)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, input, target):
        # input = torch.sigmoid(input)
        # loss = -self.alpha*(1-input)**self.gamma*(target*torch.log(input+1e-9))-\
        #        (1-self.alpha)*input**self.gamma*((1-target)*torch.log(1-input+1e-9))
        bce_loss = F.binary_cross_entropy_with_logits(input, target, reduce=False)
        pt = torch.exp(bce_loss)
        loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return loss


def cal_acc(pred,y_true):
    acc = accuracy_score(y_true,pred) #acc
    f1 = f1_score(y_true,pred,average='macro') #macro-F1
    prec = precision_score(y_true,pred,average='macro') #macro-Precision
    rec = recall_score(y_true,pred,average='macro') #macro-Recall
    return acc,f1,prec,rec

def train(args, model, device, loader, optimizer, criterion):
    model.train()

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        pred = model(batch)

        y = batch.y.view(pred.shape).to(torch.float64)
        loss = criterion(pred.double(), y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if args.log_freq and step%args.log_freq==0:
            pred_type = np.int64(torch.sigmoid(pred).detach().cpu().numpy()>0.5)
            result=cal_acc(pred_type,y.cpu().numpy())
            print(result)


def eval(args, model, device, loader):
    model.eval()
    result = []
    y_true = []
    y_test_one_hot = []
    pred_score = []
    pred_type = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch)

        y=batch.y.view(pred.shape).to(torch.long).cpu().numpy()
        pred=torch.sigmoid(pred).cpu().numpy()
        y_true.append(y)
        #y_test_one_hot.append(label_binarize(y,np.arange(86)))
        pred_score.append(pred)
        #pred_type.append(np.argmax(pred,axis=1))

    #print("before:",y_true)
    #y_true = np.array(y_true)
    y_true = np.concatenate(y_true,axis=0)
    y_test_one_hot = label_binarize(y_true,np.arange(86))
    pred_score = np.concatenate(pred_score,axis=0)
    pred_type = np.int64(pred_score>0.5)
    result = cal_acc(pred_type,y_true)


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
    parser.add_argument('--input_model_file', type=str, default='pretrained_model/MolGNet.pt',
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

    exp_path = 'runs/{}/'.format(args.dataset)
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)

    num_tasks = 1
    # set up dataset
    dataset = MoleculeDataset("data/downstream/"  + args.dataset, dataset=args.dataset, transform=None
                              )
    all_result = []
    for fold  in range(args.KFold):


        train_dataset, test_dataset = cv_random_split(dataset, fold,5)
        train_loader = DataLoaderMasking(train_dataset, batch_size=args.batch_size, shuffle=True,
                                         num_workers=args.num_workers)
        test_loader = DataLoaderMasking(test_dataset, batch_size=args.batch_size, shuffle=False,
                                        num_workers=args.num_workers)
        # set up model
        model = MolGT_graphpred(args.num_layer, args.emb_dim, args.heads, args.num_message_passing, num_tasks,
                                drop_ratio=args.dropout_ratio, graph_pooling=args.graph_pooling)
        if not args.input_model_file == "":
            model.from_pretrained(args.input_model_file)
            print('pretrained model loaded')
        else:
            print('No pretrain')

        model.to(device)
        #
        # # set up optimizer
        # # different learning rate for different part of GNN
        # #old
        # model_param_group = []
        # model_param_group.append({"params": model.gnn.parameters()})
        # if args.graph_pooling == "attention":
        #     model_param_group.append({"params": model.pool.parameters(), "lr": args.lr * args.lr_scale})
        # model_param_group.append({"params": model.graph_pred_linear.parameters(), "lr": args.lr * args.lr_scale})
        # optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)
        # print(optimizer)
        #
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay)
        # #before
        model_param_group = list(model.gnn.named_parameters())
        if args.graph_pooling == "attention":
            model_param_group += list(model.pool.named_parameters())
        model_param_group += list(model.graph_pred_linear.named_parameters())
        # optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)


        param_optimizer = [n for n in model_param_group if 'pooler' not in n[0]]
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        num_train_optimization_steps = int(len(train_dataset) / args.batch_size) * args.epochs
        print(num_train_optimization_steps)
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.lr,
                             warmup=args.warmup_proportion,
                             t_total=num_train_optimization_steps)

        if args.loss_type == 'bce':
            criterion = nn.BCEWithLogitsLoss()
        elif args.loss_type == 'softmax':
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = FocalLoss(gamma=2, alpha=0.25)

        best_result = []
        acc=0

        for epoch in range(1, args.epochs + 1):
            print("====epoch " + str(epoch))

            train(args, model, device, train_loader, optimizer,criterion)

            print("====Evaluation")
            if args.eval_train:
                train_acc = eval(args, model, device, train_loader)
            else:
                print("omit the training accuracy computation")
                train_acc = 0
            result = eval(args, model, device, test_loader)
            print(result)
            if result[1] > acc:
                acc = result[1]
                best_result=result
                torch.save(model.state_dict(), exp_path + "Fold_{}.pkl".format(fold) )
                print("save network for epoch:", epoch, acc)
            print('test metrics: acc,f1,precision,recall:', result)
        all_result.append(best_result)
        with open(exp_path+"log.txt","a+") as f:
            f.write('{}, fold {}, acc, f1, precision, recall: {}'.format(args.dataset,fold,best_result))
            f.write('\n')

    ave_result = np.mean(np.array(all_result),0)
    with open(exp_path+"log.txt", "a+") as f:
        f.write('{}, Average--acc, f1, precision, recall: {}'.format(args.dataset, ave_result))


if __name__ == "__main__":
    main()
