import os,re,glob,copy
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils import add_self_loops
from tqdm import tqdm
import sqlite3
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




class Self_loop:
    def __init__(self):
        """
        Randomly sample negative edges
        """
        pass

    def __call__(self, data,data_random=None):
        num_nodes = data.num_nodes
        data.edge_index,_ = add_self_loops(data.edge_index, num_nodes = num_nodes)
        self_loop_attr = torch.LongTensor([0, 5, 8, 10, 12]).repeat(num_nodes, 1)
        data.edge_attr = torch.cat((data.edge_attr, self_loop_attr), dim = 0)
        return data

class Add_seg_id:
    def __init__(self):
        """
        Randomly sample negative edges
        """
        pass

    def __call__(self, data,data_random=None):
        if data_random==None:
            num_nodes = data.num_nodes
            num_edges = data.num_edges
            node_seg = [0 for _ in range(num_nodes)]
            edge_seg = [0 for _ in range(num_edges)]
            data.edge_seg = torch.LongTensor(edge_seg)
            data.node_seg = torch.LongTensor(node_seg)
            return data
        else:
            num_nodes = data.num_nodes
            num_edges = data.num_edges
            node_seg = [0 for _ in range(num_nodes)]
            edge_seg = [0 for _ in range(num_edges)]

            num_nodes = data_random.num_nodes
            num_edges = data_random.num_edges
            node_seg_rd = [1 for _ in range(num_nodes)]
            edge_seg_rd = [1 for _ in range(num_edges)]

            batch = Batch.from_data_list([data,data_random])
            data = Data(x=batch.x, edge_index=batch.edge_index, edge_attr=batch.edge_attr)
            data.edge_seg = torch.LongTensor(edge_seg+edge_seg_rd)
            data.node_seg = torch.LongTensor(node_seg+node_seg_rd)
            return data

class Add_collection_node:
    def __init__(self,num_atom_type,bidirection=False):
        """
        Randomly sample negative edges
        """
        self.num_atom_type=num_atom_type
        self.bidirection = bidirection

    def __call__(self, data,data_random=None):
        num_nodes = data.num_nodes
        data.x =torch.cat((data.x,torch.tensor([[120,121,133,138,146,156,164,176]])),dim=0)
        if self.bidirection:
            dummy_edge_index = torch.LongTensor([[i for i in range(num_nodes+1)],[num_nodes for _ in range(num_nodes+1)]])
            dummy_edge_index = torch.cat((dummy_edge_index,dummy_edge_index[[1,0],:]),dim=1)
            dummy_edge_attr = torch.LongTensor([0, 5, 8, 10, 12]).repeat(2*(num_nodes+1), 1)
        else:
            dummy_edge_index = torch.LongTensor([[i for i in range(num_nodes+1)],[num_nodes for _ in range(num_nodes+1)]])
            dummy_edge_attr = torch.LongTensor([0, 5, 8, 10, 12]).repeat(num_nodes + 1, 1)

        data.edge_index = torch.cat((data.edge_index,dummy_edge_index),dim=1)
        data.edge_attr = torch.cat((data.edge_attr, dummy_edge_attr), dim=0)
        data.edge_seg = torch.cat((data.edge_seg,torch.LongTensor([2 for _ in range(dummy_edge_index.size(1))])))
        data.node_seg = torch.cat((data.node_seg,torch.LongTensor([2])))
        assert len(data.node_seg)==len(data.x)
        assert len(data.edge_seg)==len(data.edge_attr)
        data.dummy_node_indices = torch.LongTensor([num_nodes])
        return data




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



class MoleculeDataset(InMemoryDataset):
    def __init__(self,
                 root,
                 #data = None,
                 #slices = None,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 dataset='zinc250k',
                 empty=False):

        self.dataset = dataset
        self.root = root

        super(MoleculeDataset, self).__init__(root, transform, pre_transform,
                                                 pre_filter)
        self.transform, self.pre_transform, self.pre_filter = transform, pre_transform, pre_filter

        if not empty:
            self.data, self.slices = torch.load(self.processed_paths[0])


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
        data_label_list=[]

        if self.dataset == 'twosides':
            self_loop = Self_loop()
            add_seg_id = Add_seg_id()
            add_collection_node = Add_collection_node(num_atom_type=119, bidirection=False)
            conn = sqlite3.connect(self.root + "/raw/Drug_META_DDIE.db")
            drug = pd.read_sql("select * from Drug", conn)
            idToSmiles = {}
            for i in range(drug.shape[0]):
                idToSmiles[drug.loc[i][0]] = drug.loc[i][3]
            smile = drug['smile']
            positive_path = os.path.join(self.root, 'raw', 'twosides_interactions.csv')
            negative_path = os.path.join(self.root, 'raw', 'reliable_negatives.csv')
            positive = pd.read_csv(positive_path, header=None)
            negative = pd.read_csv(negative_path, header=None)
            df = pd.concat([positive, negative])
            for i in tqdm(range(df.shape[0])):
                try:
                    data1 = mol_to_graph_data_obj_complex(AllChem.MolFromSmiles(idToSmiles[df.iloc[i][0]]))
                    data2 = mol_to_graph_data_obj_complex(AllChem.MolFromSmiles(idToSmiles[df.iloc[i][1]]))
                except:
                    continue
                data1 = self_loop(data1)
                data2 = self_loop(data2)
                data = add_seg_id(data1, data2)
                data = add_collection_node(data)

                data.id = torch.tensor([i])
                data.y = torch.tensor([df.iloc[i][2]])
                data_list.append(data)

        # elif self.dataset == 'DeepDDI':
        #     self_loop=Self_loop()
        #     add_seg_id=Add_seg_id()
        #     add_collection_node=Add_collection_node(num_atom_type=119,bidirection=False)
        #     conn = sqlite3.connect(self.root + "/raw/Drug_META_DDIE.db")
        #     drug = pd.read_sql("select * from Drug", conn)
        #     idToSmiles = {}
        #     for i in range(drug.shape[0]):
        #         idToSmiles[drug.loc[i][0]] = drug.loc[i][3]
        #     smile = drug['smile']
        #     path=os.path.join(self.root,'raw','DeepDDI.csv')
        #     df = pd.read_csv(path)
        #     for i in tqdm(range(df.shape[0])):
        #         try:
        #             #print("idToSmiles",mol_to_graph_data_obj_simple(AllChem.MolFromSmiles(idToSmiles[df.iloc[i][0]])))
        #             data1 = mol_to_graph_data_obj_complex(AllChem.MolFromSmiles(idToSmiles[df.iloc[i][0]]))
        #             data2 = mol_to_graph_data_obj_complex(AllChem.MolFromSmiles(idToSmiles[df.iloc[i][1]]))
        #         except:
        #             continue
        #         data1 = self_loop(data1)
        #         data2 = self_loop(data2)
        #         data = add_seg_id(data1,data2)
        #         data = add_collection_node(data)
        #
        #         data.id = torch.tensor([i])
        #         data.y = torch.tensor([df.iloc[i][2]])
        #         data_list.append(data)
        elif self.dataset == 'biosnap':
            self_loop = Self_loop()
            add_seg_id = Add_seg_id()
            add_collection_node = Add_collection_node(num_atom_type=119, bidirection=False)

            trn = os.path.join(self.root, 'raw', 'sup_train_val.csv')
            test = os.path.join(self.root, 'raw', 'sup_test.csv')
            positive = pd.read_csv(trn)
            negative = pd.read_csv(test)
            df = pd.concat([positive, negative])

            drug1 = df['Drug1_SMILES'].values
            drug2 = df['Drug2_SMILES'].values
            labels = df['label'].values
            for i in tqdm(range(df.shape[0])):
                try:
                    data1 = mol_to_graph_data_obj_complex(AllChem.MolFromSmiles(drug1[i]))
                    data2 = mol_to_graph_data_obj_complex(AllChem.MolFromSmiles(drug2[i]))
                except:
                    continue
                data1 = self_loop(data1)
                data2 = self_loop(data2)
                data = add_seg_id(data1, data2)
                data = add_collection_node(data)
                data.y = torch.tensor([labels[i]])
                data_list.append(data)

        else:
            raise ValueError('Invalid dataset name')

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # write data_smiles_list in processed paths
        if self.dataset=='zinc_standard__agent':
            data_smiles_series = pd.Series(data_smiles_list)
            data_smiles_series.to_csv(os.path.join(self.processed_dir,
                                                   'smiles.csv'), index=False,
                                      header=False)
        else:
            label_lst = np.array(data_label_list)
            assert len(label_lst)==len(data_smiles_list)
            if label_lst.ndim==1:
                label_lst=label_lst[:,np.newaxis]
            df=pd.DataFrame({'smiles':data_smiles_list,'labels':label_lst[:,0]})
            df.to_csv(os.path.join(self.processed_dir,'smiles.csv'), index=False)


        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


# test MoleculeDataset object
if __name__ == "__main__":

    # create_all_datasets()
    from torch_geometric.data import DataLoader
    from dataloader import DataLoaderMasking
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


