import os,pickle
import numpy as np
from args import *
from model import *
from trainer import Trainer
from dataset import load_dataset_random
from utils import *
import warnings
warnings.filterwarnings("ignore")



def run_training(args):
    seed_set(args.seed)
    train_dataset, valid_dataset, test_dataset = load_dataset_random(os.path.join(args.data,args.dataset), args.dataset, args.seed)
    args.node_in_dim = train_dataset.num_node_features
    args.edge_in_dim = train_dataset.num_edge_features

    option = args.__dict__
    n_word = train_dataset.n_word
    model = CompoundProteinInteractionPrediction(n_word)
    model.from_pretrain(args.gnn)
    trainer = Trainer(option, model, train_dataset, valid_dataset, test_dataset)

    test_auc=trainer.train()
    return test_auc


def cross_validate(args):
    """k-fold cross validation"""
    # Initialize relevant variables
    init_seed = args.seed
    root_save_dir = args.exp_path
    # Run training on different random seeds for each fold
    all_scores = []
    for fold_num in range(args.num_folds):
        print('Fold {}'.format(fold_num))
        args.seed = init_seed + fold_num
        args.exp_path = os.path.join(root_save_dir, 'seed_{}'.format(args.seed))
        if not os.path.exists(args.exp_path):
            os.makedirs(args.exp_path)
        model_scores = run_training(args)
        all_scores.append(model_scores)
    all_scores = np.array(all_scores)

    # Report scores for each fold
    save_print_log('=='*20, root_save_dir)
    for fold_num, scores in enumerate(all_scores):
        msg = 'Seed {} ==> {} = {}'.format(init_seed+fold_num,args.metric,scores)
        save_print_log(msg,root_save_dir)

    # Report scores across models
    mean_score, std_score = np.nanmean(all_scores), np.nanstd(all_scores)  # average score for each model across tasks
    msg = 'Overall test {} = {} +/- {}'.format(args.metric,mean_score,std_score)
    save_print_log(msg,root_save_dir)
    return mean_score, std_score
if __name__ == '__main__':
    cross_validate(args)