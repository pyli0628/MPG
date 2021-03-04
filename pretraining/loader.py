import os,re,glob,copy
import torch
import pickle
import collections
import math
import pandas as pd
import numpy as np
import networkx as nx
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from torch.utils import data
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Batch
from itertools import repeat, product, chain
import random
from tqdm import tqdm
# from multiprocessing import Pool
from torch_geometric.data import Dataset
import os.path as osp
# from gevent.pool import Pool
# from concurrent.futures import ThreadPoolExecutor

import torch.utils.data
from torch_geometric.data import Data, Batch

import torch

class BatchMasking(Data):
    r"""A plain old python object modeling a batch of graphs as one big
    (dicconnected) graph. With :class:`torch_geometric.data.Data` being the
    base class, all its methods can also be used here.
    In addition, single graphs can be reconstructed via the assignment vector
    :obj:`batch`, which maps each node to its respective graph identifier.
    """

    def __init__(self, batch=None, **kwargs):
        super(BatchMasking, self).__init__(**kwargs)
        self.batch = batch

    @staticmethod
    def from_data_list(data_list):
        r"""Constructs a batch object from a python list holding
        :class:`torch_geometric.data.Data` objects.
        The assignment vector :obj:`batch` is created on the fly."""
        keys = [set(data.keys) for data in data_list]
        keys = list(set.union(*keys))
        assert 'batch' not in keys

        batch = BatchMasking()

        for key in keys:
            batch[key] = []
        batch.batch = []

        cumsum_node = 0
        cumsum_edge = 0

        for i, data in enumerate(data_list):
            num_nodes = data.num_nodes
            batch.batch.append(torch.full((num_nodes, ), i, dtype=torch.long))
            for key in data.keys:
                item = data[key]
                if key in ['edge_index', 'masked_atom_indices','dummy_node_indices']:
                    item = item + cumsum_node
                elif key  == 'connected_edge_indices':
                    item = item + cumsum_edge
                batch[key].append(item)

            cumsum_node += num_nodes
            cumsum_edge += data.edge_index.shape[1]

        for key in keys:
            batch[key] = torch.cat(
                batch[key], dim=data_list[0].__cat_dim__(key, batch[key][0]))
        batch.batch = torch.cat(batch.batch, dim=-1)
        return batch.contiguous()

    def cumsum(self, key, item):
        r"""If :obj:`True`, the attribute :obj:`key` with content :obj:`item`
        should be added up cumulatively before concatenated together.
        .. note::
            This method is for internal use only, and should only be overridden
            if the batch concatenation process is corrupted for a specific data
            attribute.
        """
        return key in ['edge_index', 'face', 'masked_atom_indices', 'connected_edge_indices']

    @property
    def num_graphs(self):
        """Returns the number of graphs in the batch."""
        return self.batch[-1].item() + 1


class Collater():
    def __init__(self):
        pass
    def __call__(self,batch):
        return BatchMasking.from_data_list(batch)

class DataLoaderMasking(torch.utils.data.DataLoader):
    r"""Data loader which merges data objects from a
    :class:`torch_geometric.data.dataset` to a mini-batch.
    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How may samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch (default: :obj:`True`)
    """

    def __init__(self, dataset, batch_size=1, shuffle=False, **kwargs):
        super(DataLoaderMasking, self).__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=Collater(),
            **kwargs)





allowable_features = {
    'atomic_num' : list(range(1, 122)),# 119for mask, 120 for collection
    'formal_charge' : ['unk',-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    'chirality' : ['unk',
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER
    ],
    'hybridization' : ['unk',
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2, Chem.rdchem.HybridizationType.UNSPECIFIED
    ],
    'numH' : ['unk',0, 1, 2, 3, 4, 5, 6, 7, 8],
    'implicit_valence' : ['unk',0, 1, 2, 3, 4, 5, 6],
    'degree' : ['unk',0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'isaromatic':[False,True],

    'bond_type' : ['unk',
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC
    ],
    'bond_dirs' : [ # only for double bond stereo information
        Chem.rdchem.BondDir.NONE,
        Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT
    ],
    'bond_isconjugated':[False,True],
    'bond_inring':[False,True],
    'bond_stereo': ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE","STEREOCIS", "STEREOTRANS"]
}
atom_dic = [len(allowable_features['atomic_num']),len(allowable_features['formal_charge']),len(allowable_features['chirality' ]),
            len(allowable_features['hybridization']),len(allowable_features['numH' ]),len(allowable_features['implicit_valence']),
            len(allowable_features['degree']),len(allowable_features['isaromatic'])]
bond_dic = [len(allowable_features['bond_type']),len(allowable_features['bond_dirs' ]),len(allowable_features['bond_isconjugated']),
            len(allowable_features['bond_inring']),len(allowable_features['bond_stereo'])]
atom_cumsum = np.cumsum(atom_dic)
bond_cumsum = np.cumsum(bond_dic)

def mol_to_graph_data_obj_complex(mol):
    assert mol!=None
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_feature = \
        [allowable_features['atomic_num'].index(atom.GetAtomicNum())] +\
        [allowable_features['formal_charge'].index(atom.GetFormalCharge())+atom_cumsum[0]]+\
        [allowable_features['chirality'].index(atom.GetChiralTag()) + atom_cumsum[1]]+ \
        [allowable_features['hybridization'].index(atom.GetHybridization()) + atom_cumsum[2]]+ \
        [allowable_features['numH'].index(atom.GetTotalNumHs()) + atom_cumsum[3]] + \
        [allowable_features['implicit_valence'].index(atom.GetImplicitValence()) + atom_cumsum[4]] + \
        [allowable_features['degree'].index(atom.GetDegree()) + atom_cumsum[5]] + \
        [allowable_features['isaromatic'].index(atom.GetIsAromatic()) + atom_cumsum[6]]
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)
    # bonds
    num_bond_features = 5   # bond type, bond direction
    if len(mol.GetBonds()) > 0: # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature =\
            [allowable_features['bond_type'].index(bond.GetBondType())] + \
            [allowable_features['bond_dirs'].index(bond.GetBondDir())+bond_cumsum[0]]+ \
            [allowable_features['bond_isconjugated'].index(bond.GetIsConjugated())+bond_cumsum[1]] + \
            [allowable_features['bond_inring'].index(bond.IsInRing()) + bond_cumsum[2]] + \
            [allowable_features['bond_stereo'].index(str(bond.GetStereo()))+bond_cumsum[3]]
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)
        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)
        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.tensor(np.array(edge_features_list),
                                 dtype=torch.long)
    else:   # mol has no bonds
        # print('mol has no bonds')
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return data



class PretrainDataset(InMemoryDataset):
    def __init__(self,root,rank=0,transform=None,pre_transform=None):
        super(PretrainDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[rank])
    @property
    def processed_file_names(self):
        return ['pyg_rank{}.pt'.format(i) for i in range(256)]
    @property
    def raw_file_names(self):
        return os.listdir(self.raw_dir)
    def download(self):
        pass
    def process(self):
        with open(self.raw_paths[0], 'r') as f:
            smiles_list = f.read().split('\n')
            smiles_list.pop()

        n_files=len(self.processed_file_names)
        data_list=[]
        cnt=0
        for s in tqdm(smiles_list):
            rdkit_mol = AllChem.MolFromSmiles(s)
            if rdkit_mol != None:  # ignore invalid mol objects
              data = mol_to_graph_data_obj_complex(rdkit_mol)

              rand_smi = random.sample(smiles_list,1)[0]
              rand_mol = AllChem.MolFromSmiles(rand_smi)
              rand_data = mol_to_graph_data_obj_complex(rand_mol)
              data=self.transform(data,rand_data)

              if data:
                  data_list.append(data)
              if len(data_list)==int(len(smiles_list)/n_files):
                  data, slices = self.collate(data_list)
                  torch.save((data, slices), self.processed_paths[cnt])
                  print('Rank {} saved'.format(cnt))
                  if cnt==len(self.processed_file_names)-1:
                      data_list = []
                      break
                  else:
                      cnt+=1
                      data_list=[]
        if len(data_list)>1:
            data, slices = self.collate(data_list)
            torch.save((data, slices), self.processed_paths[-1])
            print('Rank {} saved'.format(cnt))



# test MoleculeDataset object
if __name__ == "__main__":
    from util import *
    import warnings
    warnings.filterwarnings("ignore")

    transform = Compose([Random_graph(),
                         MaskAtom(num_atom_type=119, num_edge_type=5, mask_rate=0.15,
                                  mask_edge=0.15),
                         Add_seg_id(),
                         Add_collection_node(num_atom_type=119, bidirection=False)])
    dataset = PretrainDataset("data/pretraining",transform=transform)
    # # loader = DataLoader(dataset,batch_size=10,shuffle=True)
    # # for i,d in enumerate(loader):
    # #
    # #     print(d)
    # #     if i>10:
    # #         break
    #
    #
    # # data = dataset.get(0)
    # # num_node = data.num_nodes
    # # print(data)
    # # print(num_node)
    # smiles_list = \
    # pd.read_csv('/data/lpy/pretrain_dataset/dataset/' + 'bbbp' + '/processed/smiles.csv', header=None)[0].tolist()
    # train_dataset, valid_dataset, test_dataset = scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8,
    #                                                             frac_valid=0.1, frac_test=0.1)
    # # print(train_dataset.indices())
    # # print(valid_dataset.indices())
    # print(valid_dataset.get(0))
    # print(valid_dataset.get(1083))
    # print(dataset.get(0))
    # print(dataset.get(1083))
    #
    # print(len(train_dataset),len(valid_dataset),len(test_dataset))

    # train_loader = DataLoaderMasking(train_dataset, batch_size=args.batch_size, shuffle=True,
    #                                  num_workers=args.num_workers)
    # val_loader = DataLoaderMasking(valid_dataset, batch_size=args.batch_size, shuffle=False,
    #                                num_workers=args.num_workers)
    # test_loader = DataLoaderMasking(test_dataset, batch_size=args.batch_size, shuffle=False,
    #                                 num_workers=args.num_workers)


