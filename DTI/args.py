import argparse
import os
import random
import time

class Option(object):
    def __init__(self, d):
        self.__dict__ = d


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='human')
parser.add_argument("--data", type=str, default='data/downstream', help="all data dir")
parser.add_argument("--dataset", type=str, default='human')
parser.add_argument('--seed', default=33, type=int)
parser.add_argument("--gpu", type=str, nargs='+', default=None, help="CUDA device ids")
parser.add_argument("--metric", type=str, default='roc')

parser.add_argument("--hid", type=int, default=32, help="hidden size of transformer model")
parser.add_argument('--heads', default=4, type=int)
parser.add_argument('--depth', default=1, type=int)
parser.add_argument("--dropout", type=float, default=0.2)

parser.add_argument('--num_folds', default=3, type=int)
parser.add_argument('--minimize_score', default=False, action="store_true")
parser.add_argument('--num_iters',default=3,type=int)

parser.add_argument("--batch_size", type=int, default=8, help="number of batch_size")
parser.add_argument("--epochs", type=int, default=5, help="number of epochs")
parser.add_argument("--num_workers", type=int, default=0, help="dataloader worker size")
parser.add_argument("--lr", type=float, default=0.00005, help="learning rate of adam")
parser.add_argument("--weight_decay", type=float, default=1e-5)
parser.add_argument("--clip_norm", type=float, default=0.0)
parser.add_argument('--lr_scheduler_patience', default=20, type=int)
parser.add_argument('--early_stop_patience', default=-1, type=int)
parser.add_argument('--lr_decay', default=0.98, type=float)
parser.add_argument('--focalloss', default=False, action="store_true")
parser.add_argument('--single_task_mseloss', default=False)

parser.add_argument('--model_dir', default=None, type=str)
parser.add_argument('--eval', default=False, action="store_true")
parser.add_argument("--exps_dir", default='test',type=str, help="out/")
parser.add_argument('--exp_name', default=None, type=str)
parser.add_argument('--test_path', default=None, type=str)
parser.add_argument('--input_model_file', default="pretrained_model/MolGNet.pt", type=str)
parser.add_argument('--cpu', default=False, action="store_true")
parser.add_argument("--fold", type=int, default=0)
d = vars(parser.parse_args())
args = Option(d)

os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, args.gpu))


if len(args.gpu) > 1:
    args.parallel = True
    args.parallel_devices = args.gpu
else:
    args.parallel = False
    args.parallel_devices = args.gpu

if args.exp_name is None:
    args.tag = time.strftime("%m-%d-%H-%M")
else:
    args.tag = args.exp_name
args.exp_path = os.path.join(args.exps_dir, args.tag)
if not os.path.exists(args.exp_path):
    os.makedirs(args.exp_path)
args.code_file_path = os.path.abspath(__file__)
