import os,pickle,torch
from args import *
from model import *
from trainer import Trainer
from dataset import *
from utils import *
import warnings
warnings.filterwarnings("ignore")
args.node_in_dim = 8
args.edge_in_dim = 5
option = args.__dict__

for fold in range(3):
    test_dataset = TestDataset(args.model_dir+'/fold_{}/test.txt'.format(fold))
    n_word = test_dataset.n_word
    model = CompoundProteinInteractionPrediction(n_word)
    model.load_state_dict(torch.load(os.path.join(args.model_dir, 'fold_{}/model.ckpt'.format(fold))))
    trainer = Trainer(option, model, test_dataset=test_dataset)
    val_loss,roc,val_precision,val_recall=trainer.valid_iterations(mode='test')
    print('Dataset:{}, Fold:{}, ROC:{}, Precision:{}, Recall:{}'.format(args.dataset,fold,roc,val_precision,val_recall))