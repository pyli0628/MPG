import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset, Dataset
import torch_geometric.transforms as T
from rdkit import Chem, RDConfig
from rdkit.Chem import ChemicalFeatures, MolFromSmiles, AllChem
from collections import defaultdict
import pickle
import sys
from utils import *
from mol2graph import *
import re


class TestDataset(torch.utils.data.Dataset):
    def __init__(self,file_path):
        self.path = file_path
        self.datalist = []
        with open('data/downstream/3ngram_vocab', 'r') as f:
            word_dic = f.read().split('\n')
            if word_dic[-1] == '':
                word_dic.pop()

        # word_dic = ['pad'] + ['other{}'.format(n) for n in range(10)] + word_dic
        self.word_dict = {}
        for i, item in enumerate(word_dic):
            self.word_dict[item] = i
        self.n_word = len(self.word_dict)

        self.process()

    def __getitem__(self, idx):
        data = self.datalist[idx]  # attention: must self.indices()[idx]
        return data

    def __len__(self):
        return len(self.datalist)
    def split_sequence(self,sequence, ngram):
        sequence = '_' + sequence + '='
        words = [self.word_dict[sequence[i:i + ngram]]
                 for i in range(len(sequence) - ngram + 1)]
        return np.array(words)
    def process(self):

        with open(self.path,'r') as f:
            data_list = f.read().split('\n')
            if not data_list[-1]:
                data_list.pop()

        self.ngram = 3
        positive = 0
        DATALIST = []
        for no, data in enumerate(tqdm(data_list)):
            smiles, sequence, interaction = data.strip().split()

            # count positive samples
            positive+=int(interaction)

            mol = MolFromSmiles(smiles)
            if mol==None:
                continue
            data = mol_to_graph_data_dic(mol)
            data.y = torch.LongTensor([int(interaction)])
            words = self.split_sequence(sequence, self.ngram)
            data.protein = torch.LongTensor(words)
            data.pr_len = torch.LongTensor([len(words)])
            DATALIST.append(data)
        self.datalist=DATALIST



class MultiDataset(InMemoryDataset):

    def __init__(self, root, dataset, transform=None, pre_transform=None, pre_filter=None):
        self.dataset = dataset
        with open('data/downstream/3ngram_vocab', 'r') as f:
            word_dic = f.read().split('\n')
            if word_dic[-1] == '':
                word_dic.pop()
        # word_dic=['pad']+['other{}'.format(n) for n in range(10)]+word_dic
        self.word_dict = {}
        for i, item in enumerate(word_dic):
            self.word_dict[item] = i
        self.n_word = len(self.word_dict)
        super(MultiDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['data.txt']

    @property
    def processed_file_names(self):
        return ['{}.pt'.format(self.dataset)]

    def download(self):
        pass

    def split_sequence(self,sequence, ngram):
        sequence = '_' + sequence + '='
        words = [self.word_dict[sequence[i:i + ngram]]
                 for i in range(len(sequence) - ngram + 1)]
        return np.array(words)

    def dump_dictionary(self,dictionary, filename):
        with open(filename, 'wb') as f:
            pickle.dump(dict(dictionary), f)

    def process(self):

        if self.dataset=='bindingdb':
            from preprocess_dataset import process_BindingDB

        else:
            with open(self.raw_paths[0], 'r') as f:
                data_list = f.read().strip().split('\n')
            data_list = [d for d in data_list if '.' not in d.strip().split()[0]]
            N = len(data_list)

            self.ngram = 3
            positive = 0
            DATALIST = []
            for no, data in enumerate(tqdm(data_list)):
                smiles, sequence, interaction = data.strip().split()

                # count positive samples
                positive+=int(interaction)

                mol = MolFromSmiles(smiles)
                if mol==None:
                    continue
                data = mol_to_graph_data_dic(mol)
                data.y = torch.LongTensor([int(interaction)])
                words = self.split_sequence(sequence, self.ngram)
                data.protein = torch.LongTensor(words)
                data.pr_len = torch.LongTensor([len(words)])
                DATALIST.append(data)

            weights = [len(DATALIST)/(len(DATALIST)-positive),len(DATALIST)/positive]
            print(weights)

        print(len(self.word_dict))

        if self.pre_filter is not None:
            DATALIST = [data for data in DATALIST if self.pre_filter(data)]
        if self.pre_transform is not None:
            DATALIST= [self.pre_transform(data) for data in DATALIST]

        data, slices = self.collate(DATALIST)
        torch.save((data, slices), self.processed_paths[0])

def load_dataset_random(path, dataset, seed, tasks=None):
    # save_path = path + 'processed/train_valid_test_{}_seed_{}.ckpt'.format(dataset, seed)
    # if os.path.isfile(save_path):
    #     trn, val, test = torch.load(save_path)
    #     return trn, val, test
    pyg_dataset = MultiDataset(root=path, dataset=dataset)
    train_size = int(0.8 * len(pyg_dataset))
    val_size = int(0.1 * len(pyg_dataset))
    test_size = len(pyg_dataset) - train_size - val_size
    pyg_dataset = pyg_dataset.shuffle()
    trn, val, test = pyg_dataset[:train_size], \
                     pyg_dataset[train_size:(train_size + val_size)], \
                     pyg_dataset[(train_size + val_size):]
    trn.weights = 'regression task has no class weights!'
    #
    # torch.save([trn, val, test], save_path)
    return trn,val,test

if __name__=='__main__':

    # from args import *
    import torch,random,os
    # import numpy as np
    #
    # def seed_set(seed=1029):
    #     random.seed(seed)
    #     os.environ['PYTHONHASHSEED'] = str(seed)
    #     np.random.seed(seed)
    #     torch.manual_seed(seed)
    #     torch.cuda.manual_seed(seed)
    #     torch.cuda.manual_seed_all(seed)
    #     torch.backends.cudnn.deterministic = True
    #
    #
    # if args.dataset=='human':
    #     raw = os.path.join(args.data,args.dataset)+'/raw/data.txt'
    #
    #     dataset = MultiDataset(os.path.join(args.data, args.dataset), args.dataset)
    #
    #     i=0
    #     for seed in [90,34,33]:
    #         seed_set(seed)
    #         perm = torch.randperm(len(dataset)).numpy()
    #
    #         with open(raw, 'r') as f:
    #             data_list = f.read().strip().split('\n')
    #         data_list = [d for d in data_list if '.' not in d.strip().split()[0]]
    #         N = len(data_list)
    #
    #
    #         data_list=np.array(data_list)[perm]
    #
    #         train_size = int(0.8 * len(dataset))
    #         val_size = int(0.1 * len(dataset))
    #         test_size = len(dataset) - train_size - val_size
    #
    #         trn = data_list[:train_size]
    #         val = data_list[train_size:(train_size+val_size)]
    #         test = data_list[(train_size+val_size):]
    #
    #         save_dir = 'finetuned_model/DTI/{}/fold_{}/'.format(args.dataset,i)
    #         print(save_dir)
    #         with open('{}/train.txt'.format(save_dir),'w') as f:
    #             for item in trn:
    #                 f.write(item)
    #                 f.write('\n')
    #         with open('{}/valid.txt'.format(save_dir),'w') as f:
    #             for item in val:
    #                 f.write(item)
    #                 f.write('\n')
    #         with open('{}/test.txt'.format(save_dir),'w') as f:
    #             for item in test:
    #                 f.write(item)
    #                 f.write('\n')
    #         i+=1
    # if args.dataset == 'celegans':
    #     dataset = MultiDataset(os.path.join(args.data, args.dataset), args.dataset)
    #     for seed in [90,88,33]:
    #         seed_set(seed)
    #         perm = torch.randperm(len(dataset)).numpy()
    #         raw = os.path.join(args.data, args.dataset) + '/raw/data.txt'
    #         with open(raw, 'r') as f:
    #             data_list = f.read().strip().split('\n')
    #         data_list = [d for d in data_list if '.' not in d.strip().split()[0]]
    #         N = len(data_list)
    #
    #         data_list=np.array(data_list)[perm]
    #
    #         train_size = int(0.8 * len(dataset))
    #         val_size = int(0.1 * len(dataset))
    #         test_size = len(dataset) - train_size - val_size
    #
    #         trn = data_list[:train_size]
    #         val = data_list[train_size:(train_size+val_size)]
    #         test = data_list[(train_size+val_size):]
    #
    #         save_dir = 'finetuned_model/DTI/{}/seed_{}/'.format(args.dataset,seed)
    #         print(save_dir)
    #         with open('{}/train.txt'.format(save_dir),'w') as f:
    #             for item in trn:
    #                 f.write(item)
    #                 f.write('\n')
    #         with open('{}/valid.txt'.format(save_dir),'w') as f:
    #             for item in val:
    #                 f.write(item)
    #                 f.write('\n')
    #         with open('{}/test.txt'.format(save_dir),'w') as f:
    #             for item in test:
    #                 f.write(item)
    #                 f.write('\n')
    # pyg_dataset = CancerDataset('dataset/cancer', 'cancer')
    # pyg_dataset = MultiDataset2('dataset/bindingdb','bindingdb',y = 'kd')
    # train_size = int(0.8 * len(pyg_dataset))
    # val_size = int(0.1 * len(pyg_dataset))
    # test_size = len(pyg_dataset) - train_size - val_size
    # pyg_dataset = pyg_dataset.shuffle()
    # trn, val, test = pyg_dataset[:train_size], \
    #                  pyg_dataset[train_size:(train_size + val_size)], \
    #                  pyg_dataset[(train_size + val_size):]
    # print(val,test)
    # save_path = 'dataset/'+ 'processed/train_valid_test_seed.ckpt'
    # torch.save([trn, val, test], save_path)
    #
    # print(trn[0])