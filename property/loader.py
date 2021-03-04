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




class TestDataset(torch.utils.data.Dataset):
    def __init__(self,file_path,dataset,transform):
        self.path = file_path
        self.dataset = dataset
        self.transform = transform
        self.datalist = []
        self.process()

    def __getitem__(self, idx):
        data = self.datalist[idx]  # attention: must self.indices()[idx]
        data = data if self.transform is None else self.transform(data, None)
        return data

    def __len__(self):
        return len(self.datalist)

    def process(self):
        data_list = []
        if self.dataset == 'tox21':
            smiles_list, rdkit_mol_objs, labels = \
                _load_tox21_dataset(self.path)
            for i in range(len(smiles_list)):
                rdkit_mol = rdkit_mol_objs[i]
                ## convert aromatic bonds to double bonds
                # Chem.SanitizeMol(rdkit_mol,
                # sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                data = mol_to_graph_data_obj_complex(rdkit_mol)
                # manually add mol id
                data.id = torch.tensor(
                    [i])  # id here is the index of the mol in
                # the dataset
                data.y = torch.tensor(labels[i, :])
                data_list.append(data)
        elif self.dataset == 'hiv':
            smiles_list, rdkit_mol_objs, labels = \
                _load_hiv_dataset(self.path)
            for i in range(len(smiles_list)):
                rdkit_mol = rdkit_mol_objs[i]
                # # convert aromatic bonds to double bonds
                # Chem.SanitizeMol(rdkit_mol,
                #                  sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                data = mol_to_graph_data_obj_complex(rdkit_mol)
                # manually add mol id
                data.id = torch.tensor(
                    [i])  # id here is the index of the mol in
                # the dataset
                data.y = torch.tensor([labels[i]])
                data_list.append(data)
        elif self.dataset == 'bace':
            smiles_list, rdkit_mol_objs, folds, labels = \
                _load_bace_dataset(self.path)
            for i in range(len(smiles_list)):
                rdkit_mol = rdkit_mol_objs[i]
                # # convert aromatic bonds to double bonds
                # Chem.SanitizeMol(rdkit_mol,
                #                  sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                data = mol_to_graph_data_obj_complex(rdkit_mol)
                # manually add mol id
                data.id = torch.tensor(
                    [i])  # id here is the index of the mol in
                # the dataset
                data.y = torch.tensor([labels[i]])
                data.fold = torch.tensor([folds[i]])
                data_list.append(data)
        elif self.dataset == 'bbbp':
            smiles_list, rdkit_mol_objs, labels = \
                _load_bbbp_dataset(self.path)
            for i in range(len(smiles_list)):

                rdkit_mol = rdkit_mol_objs[i]
                if rdkit_mol != None:
                    # # convert aromatic bonds to double bonds
                    # Chem.SanitizeMol(rdkit_mol,
                    #                  sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                    data = mol_to_graph_data_obj_complex(rdkit_mol)
                    # manually add mol id
                    data.id = torch.tensor(
                        [i])  # id here is the index of the mol in
                    # the dataset
                    data.y = torch.tensor([labels[i]])
                    data_list.append(data)
        elif self.dataset == 'clintox':
            smiles_list, rdkit_mol_objs, labels = \
                _load_clintox_dataset(self.path)
            for i in range(len(smiles_list)):
                rdkit_mol = rdkit_mol_objs[i]
                if rdkit_mol != None:
                    # # convert aromatic bonds to double bonds
                    # Chem.SanitizeMol(rdkit_mol,
                    #                  sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                    data = mol_to_graph_data_obj_complex(rdkit_mol)
                    if data:
                        # manually add mol id
                        data.id = torch.tensor(
                            [i])  # id here is the index of the mol in
                        # the dataset
                        data.y = torch.tensor(labels[i, :])
                        data_list.append(data)

                    else:
                        print('None data')
        elif self.dataset == 'esol':
            smiles_list, rdkit_mol_objs, labels = \
                _load_esol_dataset(self.path)
            for i in range(len(smiles_list)):

                rdkit_mol = rdkit_mol_objs[i]
                # # convert aromatic bonds to double bonds
                # Chem.SanitizeMol(rdkit_mol,
                #                  sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                data = mol_to_graph_data_obj_complex(rdkit_mol)
                # manually add mol id
                if data:
                    data.id = torch.tensor(
                        [i])  # id here is the index of the mol in
                    # the dataset
                    data.y = torch.tensor([labels[i]])
                    data_list.append(data)
        elif self.dataset == 'freesolv':
            smiles_list, rdkit_mol_objs, labels = \
                _load_freesolv_dataset(self.path)
            for i in range(len(smiles_list)):

                rdkit_mol = rdkit_mol_objs[i]
                # # convert aromatic bonds to double bonds
                # Chem.SanitizeMol(rdkit_mol,
                #                  sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                data = mol_to_graph_data_obj_complex(rdkit_mol)
                if data:
                    # manually add mol id
                    data.id = torch.tensor(
                        [i])  # id here is the index of the mol in
                    # the dataset
                    data.y = torch.tensor([labels[i]])
                    data_list.append(data)
        elif self.dataset == 'lipophilicity':
            smiles_list, rdkit_mol_objs, labels = \
                _load_lipophilicity_dataset(self.path)
            for i in range(len(smiles_list)):
                rdkit_mol = rdkit_mol_objs[i]
                # # convert aromatic bonds to double bonds
                # Chem.SanitizeMol(rdkit_mol,
                #                  sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                data = mol_to_graph_data_obj_complex(rdkit_mol)
                # manually add mol id
                data.id = torch.tensor(
                    [i])  # id here is the index of the mol in
                # the dataset
                data.y = torch.tensor([labels[i]])
                data_list.append(data)
        elif self.dataset == 'muv':
            smiles_list, rdkit_mol_objs, labels = \
                _load_muv_dataset(self.raw_paths[0])
            for i in range(len(smiles_list)):
                rdkit_mol = rdkit_mol_objs[i]
                # # convert aromatic bonds to double bonds
                # Chem.SanitizeMol(rdkit_mol,
                #                  sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                data = mol_to_graph_data_obj_complex(rdkit_mol)
                # manually add mol id
                data.id = torch.tensor(
                    [i])  # id here is the index of the mol in
                # the dataset
                data.y = torch.tensor(labels[i, :])
                data_list.append(data)
        elif self.dataset == 'sider':
            smiles_list, rdkit_mol_objs, labels = \
                _load_sider_dataset(self.path)
            for i in range(len(smiles_list)):
                rdkit_mol = rdkit_mol_objs[i]
                # # convert aromatic bonds to double bonds
                # Chem.SanitizeMol(rdkit_mol,
                #                  sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                data = mol_to_graph_data_obj_complex(rdkit_mol)
                # manually add mol id
                data.id = torch.tensor(
                    [i])  # id here is the index of the mol in
                # the dataset
                data.y = torch.tensor(labels[i, :])
                data_list.append(data)
        elif self.dataset == 'toxcast':
            smiles_list, rdkit_mol_objs, labels = \
                _load_toxcast_dataset(self.path)
            for i in range(len(smiles_list)):
                rdkit_mol = rdkit_mol_objs[i]
                if rdkit_mol != None:
                    # # convert aromatic bonds to double bonds
                    # Chem.SanitizeMol(rdkit_mol,
                    #                  sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                    data = mol_to_graph_data_obj_complex(rdkit_mol)
                    # manually add mol id
                    data.id = torch.tensor(
                        [i])  # id here is the index of the mol in
                    # the dataset
                    data.y = torch.tensor(labels[i, :])
                    data_list.append(data)
        else:
            raise ValueError('Invalid dataset name')

        self.datalist = data_list








class PretrainDataset(InMemoryDataset):
    def __init__(self,root,rank=0,transform=None,pre_transform=None):
        super(PretrainDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[rank])
    @property
    def processed_file_names(self):
        return ['pyg_rank{}.pt'.format(i) for i in range(256)]
    @property
    def raw_file_names(self):
        return ['zinc_chembl_11m_smi']

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


class MoleculeDataset(InMemoryDataset):
    def __init__(self,
                 root,
                 # data = None,
                 # slices = None,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 dataset='zinc250k',
                 empty=False):
        """
        Adapted from qm9.py. Disabled the download functionality
        :param root: directory of the dataset, containing a raw and processed
        dir. The raw dir should contain the file containing the smiles, and the
        processed dir can either empty or a previously processed file
        :param dataset: name of the dataset. Currently only implemented for
        zinc250k, chembl_with_labels, tox21, hiv, bace, bbbp, clintox, esol,
        freesolv, lipophilicity, muv, pcba, sider, toxcast
        :param empty: if True, then will not load any data obj. For
        initializing empty dataset
        """
        self.dataset = dataset
        self.root = root

        super(MoleculeDataset, self).__init__(root, transform, pre_transform,
                                              pre_filter)
        self.transform, self.pre_transform, self.pre_filter = transform, pre_transform, pre_filter

        if not empty:
            self.data, self.slices = torch.load(self.processed_paths[0])

    def __getitem__(self, idx):
        r"""Gets the data object at index :obj:`idx` and transforms it (in case
        a :obj:`self.transform` is given).
        Returns a data object, if :obj:`idx` is a scalar, and a new dataset in
        case :obj:`idx` is a slicing object, *e.g.*, :obj:`[2:5]`, a LongTensor
        or a ByteTensor."""
        if isinstance(idx, int):
            # data = self.get(idx)
            data = self.get(self.indices()[idx])  # attention: must self.indices()[idx]
            idx_rd = random.randrange(self.__len__())
            data_random = self.get(self.indices()[idx_rd])  # self.get(idx_rd)
            data = data if self.transform is None else self.transform(data, data_random)
            return data
        else:
            #for split datasets
            return self.index_select(idx)

    @property
    def raw_file_names(self):
        file_name_list = os.listdir(self.raw_dir)
        # assert len(file_name_list) == 1     # currently assume we have a
        # # single raw file
        return file_name_list

    @property
    def processed_file_names(self):
        return 'geometric_data_processed_complex.pt'

    def download(self):
        raise NotImplementedError('Must indicate valid location of raw data. '
                                  'No download allowed')

    def process(self):
        data_smiles_list = []
        data_list = []
        data_label_list = []

        if self.dataset == 'zinc_standard_agent':
            input_path = self.raw_paths[0]
            input_df = pd.read_csv(input_path, sep=',', dtype='str')
            smiles_list = list(input_df['smiles'])
            zinc_id_list = list(input_df['zinc_id'])
            for i in tqdm(range(len(smiles_list))):
                s = smiles_list[i]
                # each example contains a single species
                rdkit_mol = AllChem.MolFromSmiles(s)
                if rdkit_mol != None:  # ignore invalid mol objects
                    data = mol_to_graph_data_obj_complex(rdkit_mol)
                    if data:
                        id = int(zinc_id_list[i].split('ZINC')[1].lstrip('0'))
                        data.id = torch.tensor([id])  # id here is zinc id value, stripped of
                        data_list.append(data)
                        data_smiles_list.append(smiles_list[i])
        elif self.dataset == 'embed':
            print(self.raw_paths)
            df = pd.read_csv(self.raw_paths[0])
            smiles_list = df['smiles'].values
            labels = df['scaffold_id'].values
            for i in range(len(smiles_list)):
                rdkit_mol = Chem.MolFromSmiles(smiles_list[i])
                data = mol_to_graph_data_obj_complex(rdkit_mol)
                data.y = torch.tensor([labels[i]])
                data_list.append(data)
        elif self.dataset == 'logp':
            print(self.raw_paths)
            df = pd.read_csv(self.raw_paths[0])
            smiles_list = df['smiles'].values
            labels = df['label'].values
            for i in range(len(smiles_list)):
                rdkit_mol = Chem.MolFromSmiles(smiles_list[i])
                data = mol_to_graph_data_obj_complex(rdkit_mol)
                data.y = torch.tensor([labels[i]])
                data_list.append(data)
        elif self.dataset == 'corrupt':
            print(self.raw_paths)
            df = pd.read_csv(self.raw_paths[0])
            smiles_list = df['smiles'].values

            for i in range(len(smiles_list)):
                rdkit_mol = Chem.MolFromSmiles(smiles_list[i])
                data = mol_to_graph_data_obj_complex(rdkit_mol)
                data.y = torch.tensor([1])
                data_list.append(data)

                data_c = copy.deepcopy(data)
                x = data_c.x
                idx = list(range(len(x)))
                random.shuffle(idx)
                data_c.x = x[idx]
                data_c.y = torch.tensor([0])
                data_list.append(data_c)
        elif self.dataset == 'sas':
            print(self.raw_paths)
            df = pd.read_csv(self.raw_paths[0])
            smiles_list = df['smiles'].values
            labels = df['label'].values
            for i in range(len(smiles_list)):
                rdkit_mol = Chem.MolFromSmiles(smiles_list[i])
                data = mol_to_graph_data_obj_complex(rdkit_mol)
                data.y = torch.tensor([labels[i]])
                data_list.append(data)
        elif self.dataset == 'chembl_filtered':
            ### get downstream test molecules.
            from splitters import scaffold_split
            ###
            downstream_dir = [
                'dataset/bace',
                'dataset/bbbp',
                'dataset/clintox',
                'dataset/esol',
                'dataset/freesolv',
                'dataset/hiv',
                'dataset/lipophilicity',
                'dataset/muv',
                # 'dataset/pcba/processed/smiles.csv',
                'dataset/sider',
                'dataset/tox21',
                'dataset/toxcast'
            ]

            downstream_inchi_set = set()
            for d_path in downstream_dir:
                dataset_name = d_path.split('/')[1]
                downstream_dataset = MoleculeDataset(d_path, dataset=dataset_name)
                downstream_smiles = pd.read_csv(os.path.join(d_path,
                                                             'processed', 'smiles.csv'),
                                                header=None)[0].tolist()

                assert len(downstream_dataset) == len(downstream_smiles)

                _, _, _, (train_smiles, valid_smiles, test_smiles) = scaffold_split(downstream_dataset,
                                                                                    downstream_smiles, task_idx=None,
                                                                                    null_value=0,
                                                                                    frac_train=0.8, frac_valid=0.1,
                                                                                    frac_test=0.1,
                                                                                    return_smiles=True)

                ### remove both test and validation molecules
                remove_smiles = test_smiles + valid_smiles

                downstream_inchis = []
                for smiles in remove_smiles:
                    species_list = smiles.split('.')
                    for s in species_list:  # record inchi for all species, not just
                        # largest (by default in create_standardized_mol_id if input has
                        # multiple species)
                        inchi = create_standardized_mol_id(s)
                        downstream_inchis.append(inchi)
                downstream_inchi_set.update(downstream_inchis)

            smiles_list, rdkit_mol_objs, folds, labels = \
                _load_chembl_with_labels_dataset(os.path.join(self.root, 'raw'))

            print('processing')
            for i in range(len(rdkit_mol_objs)):
                print(i)
                rdkit_mol = rdkit_mol_objs[i]
                if rdkit_mol != None:
                    # # convert aromatic bonds to double bonds
                    # Chem.SanitizeMol(rdkit_mol,
                    #                  sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                    mw = Descriptors.MolWt(rdkit_mol)
                    if 50 <= mw <= 900:
                        inchi = create_standardized_mol_id(smiles_list[i])
                        if inchi != None and inchi not in downstream_inchi_set:
                            data = mol_to_graph_data_obj_simple(rdkit_mol)
                            # manually add mol id
                            data.id = torch.tensor(
                                [i])  # id here is the index of the mol in
                            # the dataset
                            data.y = torch.tensor(labels[i, :])
                            # fold information
                            if i in folds[0]:
                                data.fold = torch.tensor([0])
                            elif i in folds[1]:
                                data.fold = torch.tensor([1])
                            else:
                                data.fold = torch.tensor([2])
                            data_list.append(data)
                            data_smiles_list.append(smiles_list[i])

        elif self.dataset == 'tox21':
            smiles_list, rdkit_mol_objs, labels = \
                _load_tox21_dataset(self.raw_paths[0])
            for i in range(len(smiles_list)):
                rdkit_mol = rdkit_mol_objs[i]
                ## convert aromatic bonds to double bonds
                # Chem.SanitizeMol(rdkit_mol,
                # sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                data = mol_to_graph_data_obj_complex(rdkit_mol)
                # manually add mol id
                data.id = torch.tensor(
                    [i])  # id here is the index of the mol in
                # the dataset
                data.y = torch.tensor(labels[i, :])
                data_list.append(data)
                data_smiles_list.append(smiles_list[i])
                data_label_list.append(labels[i])

        elif self.dataset == 'hiv':
            smiles_list, rdkit_mol_objs, labels = \
                _load_hiv_dataset(self.raw_paths[0])
            for i in range(len(smiles_list)):
                rdkit_mol = rdkit_mol_objs[i]
                # # convert aromatic bonds to double bonds
                # Chem.SanitizeMol(rdkit_mol,
                #                  sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                data = mol_to_graph_data_obj_complex(rdkit_mol)
                # manually add mol id
                data.id = torch.tensor(
                    [i])  # id here is the index of the mol in
                # the dataset
                data.y = torch.tensor([labels[i]])
                data_list.append(data)
                data_smiles_list.append(smiles_list[i])
                data_label_list.append(labels[i])

        elif self.dataset == 'bace':
            smiles_list, rdkit_mol_objs, folds, labels = \
                _load_bace_dataset(self.raw_paths[0])
            for i in range(len(smiles_list)):
                rdkit_mol = rdkit_mol_objs[i]
                # # convert aromatic bonds to double bonds
                # Chem.SanitizeMol(rdkit_mol,
                #                  sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                data = mol_to_graph_data_obj_complex(rdkit_mol)
                # manually add mol id
                data.id = torch.tensor(
                    [i])  # id here is the index of the mol in
                # the dataset
                data.y = torch.tensor([labels[i]])
                data.fold = torch.tensor([folds[i]])
                data_list.append(data)
                data_smiles_list.append(smiles_list[i])
                data_label_list.append(labels[i])

        elif self.dataset == 'bbbp':
            smiles_list, rdkit_mol_objs, labels = \
                _load_bbbp_dataset(self.raw_paths[0])
            for i in range(len(smiles_list)):

                rdkit_mol = rdkit_mol_objs[i]
                if rdkit_mol != None:
                    # # convert aromatic bonds to double bonds
                    # Chem.SanitizeMol(rdkit_mol,
                    #                  sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                    data = mol_to_graph_data_obj_complex(rdkit_mol)
                    # manually add mol id
                    data.id = torch.tensor(
                        [i])  # id here is the index of the mol in
                    # the dataset
                    data.y = torch.tensor([labels[i]])
                    data_list.append(data)
                    data_smiles_list.append(smiles_list[i])
                    data_label_list.append(labels[i])

        elif self.dataset == 'clintox':
            smiles_list, rdkit_mol_objs, labels = \
                _load_clintox_dataset(self.raw_paths[0])
            for i in range(len(smiles_list)):
                rdkit_mol = rdkit_mol_objs[i]
                if rdkit_mol != None:
                    # # convert aromatic bonds to double bonds
                    # Chem.SanitizeMol(rdkit_mol,
                    #                  sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                    data = mol_to_graph_data_obj_complex(rdkit_mol)
                    if data:
                        # manually add mol id
                        data.id = torch.tensor(
                            [i])  # id here is the index of the mol in
                        # the dataset
                        data.y = torch.tensor(labels[i, :])
                        data_list.append(data)
                        data_smiles_list.append(smiles_list[i])
                        data_label_list.append(labels[i])
                    else:
                        print('None data')

        elif self.dataset == 'esol':
            smiles_list, rdkit_mol_objs, labels = \
                _load_esol_dataset(self.raw_paths[0])
            for i in range(len(smiles_list)):

                rdkit_mol = rdkit_mol_objs[i]
                # # convert aromatic bonds to double bonds
                # Chem.SanitizeMol(rdkit_mol,
                #                  sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                data = mol_to_graph_data_obj_complex(rdkit_mol)
                # manually add mol id
                if data:
                    data.id = torch.tensor(
                        [i])  # id here is the index of the mol in
                    # the dataset
                    data.y = torch.tensor([labels[i]])
                    data_list.append(data)
                    data_smiles_list.append(smiles_list[i])
                    data_label_list.append(labels[i])

        elif self.dataset == 'freesolv':
            smiles_list, rdkit_mol_objs, labels = \
                _load_freesolv_dataset(self.raw_paths[0])
            for i in range(len(smiles_list)):

                rdkit_mol = rdkit_mol_objs[i]
                # # convert aromatic bonds to double bonds
                # Chem.SanitizeMol(rdkit_mol,
                #                  sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                data = mol_to_graph_data_obj_complex(rdkit_mol)
                if data:
                    # manually add mol id
                    data.id = torch.tensor(
                        [i])  # id here is the index of the mol in
                    # the dataset
                    data.y = torch.tensor([labels[i]])
                    data_list.append(data)
                    data_smiles_list.append(smiles_list[i])
                    data_label_list.append(labels[i])

        elif self.dataset == 'lipophilicity':
            smiles_list, rdkit_mol_objs, labels = \
                _load_lipophilicity_dataset(self.raw_paths[0])
            for i in range(len(smiles_list)):
                rdkit_mol = rdkit_mol_objs[i]
                # # convert aromatic bonds to double bonds
                # Chem.SanitizeMol(rdkit_mol,
                #                  sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                data = mol_to_graph_data_obj_complex(rdkit_mol)
                # manually add mol id
                data.id = torch.tensor(
                    [i])  # id here is the index of the mol in
                # the dataset
                data.y = torch.tensor([labels[i]])
                data_list.append(data)
                data_smiles_list.append(smiles_list[i])
                data_label_list.append(labels[i])

        elif self.dataset == 'muv':
            smiles_list, rdkit_mol_objs, labels = \
                _load_muv_dataset(self.raw_paths[0])
            for i in range(len(smiles_list)):
                rdkit_mol = rdkit_mol_objs[i]
                # # convert aromatic bonds to double bonds
                # Chem.SanitizeMol(rdkit_mol,
                #                  sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                data = mol_to_graph_data_obj_complex(rdkit_mol)
                # manually add mol id
                data.id = torch.tensor(
                    [i])  # id here is the index of the mol in
                # the dataset
                data.y = torch.tensor(labels[i, :])
                data_list.append(data)
                data_smiles_list.append(smiles_list[i])
                data_label_list.append(labels[i])
        # elif self.dataset == ''
        elif self.dataset == 'sider':
            smiles_list, rdkit_mol_objs, labels = \
                _load_sider_dataset(self.raw_paths[0])
            for i in range(len(smiles_list)):
                rdkit_mol = rdkit_mol_objs[i]
                # # convert aromatic bonds to double bonds
                # Chem.SanitizeMol(rdkit_mol,
                #                  sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                data = mol_to_graph_data_obj_complex(rdkit_mol)
                # manually add mol id
                data.id = torch.tensor(
                    [i])  # id here is the index of the mol in
                # the dataset
                data.y = torch.tensor(labels[i, :])
                data_list.append(data)
                data_smiles_list.append(smiles_list[i])
                data_label_list.append(labels[i])

        elif self.dataset == 'toxcast':
            smiles_list, rdkit_mol_objs, labels = \
                _load_toxcast_dataset(self.raw_paths[0])
            for i in range(len(smiles_list)):
                rdkit_mol = rdkit_mol_objs[i]
                if rdkit_mol != None:
                    # # convert aromatic bonds to double bonds
                    # Chem.SanitizeMol(rdkit_mol,
                    #                  sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                    data = mol_to_graph_data_obj_complex(rdkit_mol)
                    # manually add mol id
                    data.id = torch.tensor(
                        [i])  # id here is the index of the mol in
                    # the dataset
                    data.y = torch.tensor(labels[i, :])
                    data_list.append(data)
                    data_smiles_list.append(smiles_list[i])
                    data_label_list.append(labels[i])


        else:
            raise ValueError('Invalid dataset name')

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # write data_smiles_list in processed paths
        if self.dataset == 'zinc_standard__agent':
            data_smiles_series = pd.Series(data_smiles_list)
            data_smiles_series.to_csv(os.path.join(self.processed_dir,
                                                   'smiles.csv'), index=False,
                                      header=False)
        else:
            label_lst = np.array(data_label_list)
            assert len(label_lst) == len(data_smiles_list)
            if label_lst.ndim == 1:
                label_lst = label_lst[:, np.newaxis]
            df = pd.DataFrame({'smiles': data_smiles_list, 'labels': label_lst[:, 0]})
            df.to_csv(os.path.join(self.processed_dir, 'smiles.csv'), index=False)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def _load_tox21_dataset(input_path):
    """

    :param input_path:
    :return: list of smiles, list of rdkit mol obj, np.array containing the
    labels
    """
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    tasks = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
       'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']
    labels = input_df[tasks]
    # convert 0 to -1
    labels = labels.replace(0, -1)
    # convert nan to 0
    labels = labels.fillna(0)
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    return smiles_list, rdkit_mol_objs_list, labels.values

def _load_hiv_dataset(input_path):
    """
    :param input_path:
    :return: list of smiles, list of rdkit mol obj, np.array containing the
    labels
    """
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    labels = input_df['HIV_active']
    # convert 0 to -1
    labels = labels.replace(0, -1)
    # there are no nans
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    return smiles_list, rdkit_mol_objs_list, labels.values

def _load_bace_dataset(input_path):
    """

    :param input_path:
    :return: list of smiles, list of rdkit mol obj, np.array
    containing indices for each of the 3 folds, np.array containing the
    labels
    """
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['mol']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    labels = input_df['Class']
    # convert 0 to -1
    labels = labels.replace(0, -1)
    # there are no nans
    folds = input_df['Model']
    folds = folds.replace('Train', 0)   # 0 -> train
    folds = folds.replace('Valid', 1)   # 1 -> valid
    folds = folds.replace('Test', 2)    # 2 -> test
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    assert len(smiles_list) == len(folds)
    return smiles_list, rdkit_mol_objs_list, folds.values, labels.values

def _load_bbbp_dataset(input_path):
    """

    :param input_path:
    :return: list of smiles, list of rdkit mol obj, np.array containing the
    labels
    """
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]

    preprocessed_rdkit_mol_objs_list = [m if m != None else None for m in
                                                          rdkit_mol_objs_list]
    preprocessed_smiles_list = [AllChem.MolToSmiles(m) if m != None else
                                None for m in preprocessed_rdkit_mol_objs_list]
    labels = input_df['p_np']
    # convert 0 to -1
    labels = labels.replace(0, -1)
    # there are no nans
    assert len(smiles_list) == len(preprocessed_rdkit_mol_objs_list)
    assert len(smiles_list) == len(preprocessed_smiles_list)
    assert len(smiles_list) == len(labels)
    return preprocessed_smiles_list, preprocessed_rdkit_mol_objs_list, \
           labels.values

def _load_clintox_dataset(input_path):
    """

    :param input_path:
    :return: list of smiles, list of rdkit mol obj, np.array containing the
    labels
    """
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]

    preprocessed_rdkit_mol_objs_list = [m if m != None else None for m in
                                        rdkit_mol_objs_list]
    preprocessed_smiles_list = [AllChem.MolToSmiles(m) if m != None else
                                None for m in preprocessed_rdkit_mol_objs_list]
    tasks = ['FDA_APPROVED', 'CT_TOX']
    labels = input_df[tasks]
    # convert 0 to -1
    labels = labels.replace(0, -1)
    # there are no nans
    assert len(smiles_list) == len(preprocessed_rdkit_mol_objs_list)
    assert len(smiles_list) == len(preprocessed_smiles_list)
    assert len(smiles_list) == len(labels)
    return preprocessed_smiles_list, preprocessed_rdkit_mol_objs_list, \
           labels.values
# input_path = 'dataset/clintox/raw/clintox.csv'
# smiles_list, rdkit_mol_objs_list, labels = _load_clintox_dataset(input_path)

def _load_esol_dataset(input_path):
    """

    :param input_path:
    :return: list of smiles, list of rdkit mol obj, np.array containing the
    labels (regression task)
    """
    # NB: some examples have multiple species
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    labels = input_df['measured log solubility in mols per litre']
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    return smiles_list, rdkit_mol_objs_list, labels.values
# input_path = 'dataset/esol/raw/delaney-processed.csv'
# smiles_list, rdkit_mol_objs_list, labels = _load_esol_dataset(input_path)

def _load_freesolv_dataset(input_path):
    """

    :param input_path:
    :return: list of smiles, list of rdkit mol obj, np.array containing the
    labels (regression task)
    """
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    labels = input_df['expt']
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    return smiles_list, rdkit_mol_objs_list, labels.values


def _load_lipophilicity_dataset(input_path):
    """

    :param input_path:
    :return: list of smiles, list of rdkit mol obj, np.array containing the
    labels (regression task)
    """
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    labels = input_df['exp']
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    return smiles_list, rdkit_mol_objs_list, labels.values


def _load_muv_dataset(input_path):
    """

    :param input_path:
    :return: list of smiles, list of rdkit mol obj, np.array containing the
    labels
    """
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    tasks = ['MUV-466', 'MUV-548', 'MUV-600', 'MUV-644', 'MUV-652', 'MUV-689',
       'MUV-692', 'MUV-712', 'MUV-713', 'MUV-733', 'MUV-737', 'MUV-810',
       'MUV-832', 'MUV-846', 'MUV-852', 'MUV-858', 'MUV-859']
    labels = input_df[tasks]
    # convert 0 to -1
    labels = labels.replace(0, -1)
    # convert nan to 0
    labels = labels.fillna(0)
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    return smiles_list, rdkit_mol_objs_list, labels.values

def _load_sider_dataset(input_path):
    """

    :param input_path:
    :return: list of smiles, list of rdkit mol obj, np.array containing the
    labels
    """
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    tasks = ['Hepatobiliary disorders',
       'Metabolism and nutrition disorders', 'Product issues', 'Eye disorders',
       'Investigations', 'Musculoskeletal and connective tissue disorders',
       'Gastrointestinal disorders', 'Social circumstances',
       'Immune system disorders', 'Reproductive system and breast disorders',
       'Neoplasms benign, malignant and unspecified (incl cysts and polyps)',
       'General disorders and administration site conditions',
       'Endocrine disorders', 'Surgical and medical procedures',
       'Vascular disorders', 'Blood and lymphatic system disorders',
       'Skin and subcutaneous tissue disorders',
       'Congenital, familial and genetic disorders',
       'Infections and infestations',
       'Respiratory, thoracic and mediastinal disorders',
       'Psychiatric disorders', 'Renal and urinary disorders',
       'Pregnancy, puerperium and perinatal conditions',
       'Ear and labyrinth disorders', 'Cardiac disorders',
       'Nervous system disorders',
       'Injury, poisoning and procedural complications']
    labels = input_df[tasks]
    # convert 0 to -1
    labels = labels.replace(0, -1)
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    return smiles_list, rdkit_mol_objs_list, labels.values

def _load_toxcast_dataset(input_path):
    """

    :param input_path:
    :return: list of smiles, list of rdkit mol obj, np.array containing the
    labels
    """
    # NB: some examples have multiple species, some example smiles are invalid
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    # Some smiles could not be successfully converted
    # to rdkit mol object so them to None
    preprocessed_rdkit_mol_objs_list = [m if m != None else None for m in
                                        rdkit_mol_objs_list]
    preprocessed_smiles_list = [AllChem.MolToSmiles(m) if m != None else
                                None for m in preprocessed_rdkit_mol_objs_list]
    tasks = list(input_df.columns)[1:]
    labels = input_df[tasks]
    # convert 0 to -1
    labels = labels.replace(0, -1)
    # convert nan to 0
    labels = labels.fillna(0)
    assert len(smiles_list) == len(preprocessed_rdkit_mol_objs_list)
    assert len(smiles_list) == len(preprocessed_smiles_list)
    assert len(smiles_list) == len(labels)
    return preprocessed_smiles_list, preprocessed_rdkit_mol_objs_list, \
           labels.values

def _load_chembl_with_labels_dataset(root_path):
    """
    Data from 'Large-scale comparison of machine learning methods for drug target prediction on ChEMBL'
    :param root_path: path to the folder containing the reduced chembl dataset
    :return: list of smiles, preprocessed rdkit mol obj list, list of np.array
    containing indices for each of the 3 folds, np.array containing the labels
    """
    # adapted from https://github.com/ml-jku/lsc/blob/master/pythonCode/lstm/loadData.py
    # first need to download the files and unzip:
    # wget http://bioinf.jku.at/research/lsc/chembl20/dataPythonReduced.zip
    # unzip and rename to chembl_with_labels
    # wget http://bioinf.jku.at/research/lsc/chembl20/dataPythonReduced/chembl20Smiles.pckl
    # into the dataPythonReduced directory
    # wget http://bioinf.jku.at/research/lsc/chembl20/dataPythonReduced/chembl20LSTM.pckl

    # 1. load folds and labels
    f=open(os.path.join(root_path, 'folds0.pckl'), 'rb')
    folds=pickle.load(f)
    f.close()

    f=open(os.path.join(root_path, 'labelsHard.pckl'), 'rb')
    targetMat=pickle.load(f)
    sampleAnnInd=pickle.load(f)
    targetAnnInd=pickle.load(f)
    f.close()

    targetMat=targetMat
    targetMat=targetMat.copy().tocsr()
    targetMat.sort_indices()
    targetAnnInd=targetAnnInd
    targetAnnInd=targetAnnInd-targetAnnInd.min()

    folds=[np.intersect1d(fold, sampleAnnInd.index.values).tolist() for fold in folds]
    targetMatTransposed=targetMat[sampleAnnInd[list(chain(*folds))]].T.tocsr()
    targetMatTransposed.sort_indices()
    # # num positive examples in each of the 1310 targets
    trainPosOverall=np.array([np.sum(targetMatTransposed[x].data > 0.5) for x in range(targetMatTransposed.shape[0])])
    # # num negative examples in each of the 1310 targets
    trainNegOverall=np.array([np.sum(targetMatTransposed[x].data < -0.5) for x in range(targetMatTransposed.shape[0])])
    # dense array containing the labels for the 456331 molecules and 1310 targets
    denseOutputData=targetMat.A # possible values are {-1, 0, 1}

    # 2. load structures
    f=open(os.path.join(root_path, 'chembl20LSTM.pckl'), 'rb')
    rdkitArr=pickle.load(f)
    f.close()

    assert len(rdkitArr) == denseOutputData.shape[0]
    assert len(rdkitArr) == len(folds[0]) + len(folds[1]) + len(folds[2])

    preprocessed_rdkitArr = []
    print('preprocessing')
    for i in range(len(rdkitArr)):
        print(i)
        m = rdkitArr[i]
        if m == None:
            preprocessed_rdkitArr.append(None)
        else:
            mol_species_list = split_rdkit_mol_obj(m)
            if len(mol_species_list) == 0:
                preprocessed_rdkitArr.append(None)
            else:
                largest_mol = get_largest_mol(mol_species_list)
                if len(largest_mol.GetAtoms()) <= 2:
                    preprocessed_rdkitArr.append(None)
                else:
                    preprocessed_rdkitArr.append(largest_mol)

    assert len(preprocessed_rdkitArr) == denseOutputData.shape[0]

    smiles_list = [AllChem.MolToSmiles(m) if m != None else None for m in
                   preprocessed_rdkitArr]   # bc some empty mol in the
    # rdkitArr zzz...

    assert len(preprocessed_rdkitArr) == len(smiles_list)

    return smiles_list, preprocessed_rdkitArr, folds, denseOutputData
# root_path = 'dataset/chembl_with_labels'

# test MoleculeDataset object
if __name__ == "__main__":

    # create_all_datasets()
    from util import *
    from splitters import *
    import warnings
    warnings.filterwarnings("ignore")

    transform = Compose([Random_graph(),
                         MaskAtom(num_atom_type=119, num_edge_type=5, mask_rate=0.15,
                                  mask_edge=0.15),
                         Add_seg(),
                         Add_collection_node(num_atom_type=119, bidirection=False)])
    dataset = PretrainDataset("data/pretraining",transform=transform)

