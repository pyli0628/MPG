import math,copy,random
import numpy as np
from rdkit import Chem
from rdkit.Chem import MolFromSmiles,MolToSmiles,CanonSmiles, AllChem
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj
from torch_geometric.utils import add_self_loops

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
allowable_features_dic={}
for key,value in allowable_features.items():
    value_id = list(range(len(value)))
    value_dic = dict(zip(value,value_id))
    allowable_features_dic[key]=value_dic

atom_dic = [len(allowable_features['atomic_num']),len(allowable_features['formal_charge']),len(allowable_features['chirality' ]),
            len(allowable_features['hybridization']),len(allowable_features['numH' ]),len(allowable_features['implicit_valence']),
            len(allowable_features['degree']),len(allowable_features['isaromatic'])]
bond_dic = [len(allowable_features['bond_type']),len(allowable_features['bond_dirs' ]),len(allowable_features['bond_isconjugated']),
            len(allowable_features['bond_inring']),len(allowable_features['bond_stereo'])]
atom_cumsum = np.cumsum(atom_dic)
bond_cumsum = np.cumsum(bond_dic)


def Add_Loop(data):
    num_nodes = data.num_nodes
    data.edge_index,_ = add_self_loops(data.edge_index, num_nodes = num_nodes)
    self_loop_attr = torch.LongTensor([0, 5, 8, 10, 12]).repeat(num_nodes, 1)
    data.edge_attr = torch.cat((data.edge_attr, self_loop_attr), dim = 0)
    return data


def Add_Seg(data):
    num_nodes = data.num_nodes
    num_edges = data.num_edges
    node_seg = [0 for _ in range(num_nodes)]
    edge_seg = [0 for _ in range(num_edges)]
    data.edge_seg = torch.LongTensor(edge_seg)
    data.node_seg = torch.LongTensor(node_seg)
    return data

def Add_collection_node(data):
    num_nodes = data.num_nodes
    data.x =torch.cat((data.x,torch.tensor([[120,121,133,138,146,156,164,176]])),dim=0)

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


def mol_to_graph_data_dic(mol):
    assert mol != None
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_feature = \
            [allowable_features_dic['atomic_num'].get(atom.GetAtomicNum(),119)] + \
            [allowable_features_dic['formal_charge'].get(atom.GetFormalCharge(),0) + atom_cumsum[0]] + \
            [allowable_features_dic['chirality'].get(atom.GetChiralTag(),0) + atom_cumsum[1]] + \
            [allowable_features_dic['hybridization'].get(atom.GetHybridization(),0) + atom_cumsum[2]] + \
            [allowable_features_dic['numH'].get(atom.GetTotalNumHs(),0) + atom_cumsum[3]] + \
            [allowable_features_dic['implicit_valence'].get(atom.GetImplicitValence(),0) + atom_cumsum[4]] + \
            [allowable_features_dic['degree'].get(atom.GetDegree(),0) + atom_cumsum[5]] + \
            [allowable_features_dic['isaromatic'].get(atom.GetIsAromatic(),0) + atom_cumsum[6]]

        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)
    assert x.size(1)==8
    # bonds
    num_bond_features = 5  # bond type, bond direction
    if len(mol.GetBonds()) > 0:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = \
                [allowable_features['bond_type'].index(bond.GetBondType())] + \
                [allowable_features['bond_dirs'].index(bond.GetBondDir()) + bond_cumsum[0]] + \
                [allowable_features['bond_isconjugated'].index(bond.GetIsConjugated()) + bond_cumsum[1]] + \
                [allowable_features['bond_inring'].index(bond.IsInRing()) + bond_cumsum[2]] + \
                [allowable_features['bond_stereo'].index(str(bond.GetStereo())) + bond_cumsum[3]]
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)
        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)
        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.tensor(np.array(edge_features_list),
                                 dtype=torch.long)
    else:  # mol has no bonds
        # print('mol has no bonds')
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data = Add_Loop(data)
    data = Add_Seg(data)
    data = Add_collection_node(data)
    return data
