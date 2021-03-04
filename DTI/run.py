import os,pickle,torch
from args import *
from model import *
from trainer import Trainer
from dataset import TestDataset
from utils import *
import warnings
warnings.filterwarnings("ignore")
args.node_in_dim = 8
args.edge_in_dim = 5
fold=args.fold
args.exp_path = os.path.join(args.exp_path, 'fold_{}'.format(fold))
if not os.path.exists(args.exp_path):
    os.makedirs(args.exp_path)
option = args.__dict__
train_dataset = TestDataset(args.model_dir+'/fold_{}/train.txt'.format(fold))
val_dataset = TestDataset(args.model_dir+'/fold_{}/valid.txt'.format(fold))
test_dataset = TestDataset(args.model_dir+'/fold_{}/test.txt'.format(fold))
n_word = test_dataset.n_word
model = CompoundProteinInteractionPrediction(n_word)
model.from_pretrain(args.gnn)
trainer = Trainer(option, model, train_dataset, val_dataset, test_dataset)
test_auc = trainer.train()