import random
import os
import torch
import numpy as np
import pandas as pd
from torch_geometric.utils import remove_self_loops
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


def save_print_log(msg, save_dir=None, show=True):
    with open(save_dir + '/log.txt', 'a+') as f:
        f.write(msg + '\n')
        if show:
            print(msg)

def seed_set(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def onehot_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(
            x, allowable_set))
    return [x == s for s in allowable_set]


def onehot_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]


def split_multi_label_containNan(df, tasks, seed):
    weights = []
    random_seed = seed
    for i, task in enumerate(tasks):
        negative_df = df[df[task] == 0][["smiles", task]]
        positive_df = df[df[task] == 1][["smiles", task]]
        negative_test = negative_df.sample(frac=1 / 10, random_state=random_seed)
        negative_valid = negative_df.drop(negative_test.index).sample(frac=1 / 9, random_state=random_seed)
        negative_train = negative_df.drop(negative_test.index).drop(negative_valid.index)

        positive_test = positive_df.sample(frac=1 / 10, random_state=random_seed)
        positive_valid = positive_df.drop(positive_test.index).sample(frac=1 / 9, random_state=random_seed)
        positive_train = positive_df.drop(positive_test.index).drop(positive_valid.index)

        weights.append([(positive_train.shape[0] + negative_train.shape[0]) / negative_train.shape[0], \
                        (positive_train.shape[0] + negative_train.shape[0]) / positive_train.shape[0]])
        train_df_new = pd.concat([negative_train, positive_train])
        valid_df_new = pd.concat([negative_valid, positive_valid])
        test_df_new = pd.concat([negative_test, positive_test])

        if i == 0:
            train_df = train_df_new
            test_df = test_df_new
            valid_df = valid_df_new
        else:
            train_df = pd.merge(train_df, train_df_new, on='smiles', how='outer')
            test_df = pd.merge(test_df, test_df_new, on='smiles', how='outer')
            valid_df = pd.merge(valid_df, valid_df_new, on='smiles', how='outer')
    return train_df, valid_df, test_df, weights


class ScaffoldGenerator(object):
    """
    Generate molecular scaffolds.

    Parameters
    ----------
    include_chirality : : bool, optional (default False)
        Include chirality in scaffolds.
    """

    def __init__(self, include_chirality=False):
        self.include_chirality = include_chirality

    def get_scaffold(self, mol):
        """
        Get Murcko scaffolds for molecules.

        Murcko scaffolds are described in DOI: 10.1021/jm9602928. They are
        essentially that part of the molecule consisting of rings and the
        linker atoms between them.

        Parameters
        ----------
        mols : array_like
            Molecules.
        """
        return MurckoScaffold.MurckoScaffoldSmiles(
            mol=mol, includeChirality=self.include_chirality)


def generate_scaffold(smiles, include_chirality=False):
    """Compute the Bemis-Murcko scaffold for a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    engine = ScaffoldGenerator(include_chirality=include_chirality)
    scaffold = engine.get_scaffold(mol)
    return scaffold


def split(scaffolds_dict, smiles_tasks_df, tasks, weights, sample_size, random_seed=0):
    count = 0
    minor_count = 0
    minor_class = np.argmax(weights[0])  # weights are inverse of the ratio
    minor_ratio = 1 / weights[0][minor_class]
    optimal_count = 0.1 * len(smiles_tasks_df)
    while (count < optimal_count * 0.9 or count > optimal_count * 1.1) \
            or (minor_count < minor_ratio * optimal_count * 0.9 \
                or minor_count > minor_ratio * optimal_count * 1.1):
        random_seed += 1
        random.seed(random_seed)
        scaffold = random.sample(list(scaffolds_dict.keys()), sample_size)
        count = sum([len(scaffolds_dict[scaffold]) for scaffold in scaffold])
        index = [index for scaffold in scaffold for index in scaffolds_dict[scaffold]]
        minor_count = len(smiles_tasks_df.iloc[index, :][smiles_tasks_df[tasks[0]] == minor_class])
    #     print(random)
    return scaffold, index


def scaffold_randomized_spliting(smiles_tasks_df, tasks=['HIV_active'], random_seed=8):
    weights = []
    for i, task in enumerate(tasks):
        negative_df = smiles_tasks_df[smiles_tasks_df[task] == 0][["smiles", task]]
        positive_df = smiles_tasks_df[smiles_tasks_df[task] == 1][["smiles", task]]
        weights.append([(positive_df.shape[0] + negative_df.shape[0]) / negative_df.shape[0], \
                        (positive_df.shape[0] + negative_df.shape[0]) / positive_df.shape[0]])
    print('The dataset weights are', weights)
    print('generating scaffold......')
    scaffold_list = []
    all_scaffolds_dict = {}
    for index, smiles in enumerate(smiles_tasks_df['smiles']):
        scaffold = generate_scaffold(smiles)
        scaffold_list.append(scaffold)
        if scaffold not in all_scaffolds_dict:
            all_scaffolds_dict[scaffold] = [index]
        else:
            all_scaffolds_dict[scaffold].append(index)
    #     smiles_tasks_df['scaffold'] = scaffold_list

    samples_size = int(len(all_scaffolds_dict.keys()) * 0.1)
    test_scaffold, test_index = split(all_scaffolds_dict, smiles_tasks_df, tasks, weights, samples_size,
                                      random_seed=random_seed)
    training_scaffolds_dict = {x: all_scaffolds_dict[x] for x in all_scaffolds_dict.keys() if x not in test_scaffold}
    valid_scaffold, valid_index = split(training_scaffolds_dict, smiles_tasks_df, tasks, weights, samples_size,
                                        random_seed=random_seed)

    training_scaffolds_dict = {x: training_scaffolds_dict[x] for x in training_scaffolds_dict.keys() if
                               x not in valid_scaffold}
    train_index = []
    for ele in training_scaffolds_dict.values():
        train_index += ele
    assert len(train_index) + len(valid_index) + len(test_index) == len(smiles_tasks_df)

    return train_index, valid_index, test_index, weights


class Complete(object):
    def __call__(self, data):
        device = data.edge_index.device

        row = torch.arange(data.num_nodes, dtype=torch.long, device=device)
        col = torch.arange(data.num_nodes, dtype=torch.long, device=device)

        row = row.view(-1, 1).repeat(1, data.num_nodes).view(-1)
        col = col.repeat(data.num_nodes)
        edge_index = torch.stack([row, col], dim=0)

        edge_attr = None
        if data.edge_attr is not None:
            idx = data.edge_index[0] * data.num_nodes + data.edge_index[1]
            size = list(data.edge_attr.size())
            size[0] = data.num_nodes * data.num_nodes
            edge_attr = data.edge_attr.new_zeros(size)
            edge_attr[idx] = data.edge_attr

        edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
        data.edge_attr = edge_attr
        data.edge_index = edge_index

        return data


def get_latest_ckpt(file_dir='./ckpt/'):
    filelist = os.listdir(file_dir)
    filelist.sort(key=lambda fn: os.path.getmtime(file_dir + fn) if not os.path.isdir(file_dir + fn) else 0)
    print('The latest ckpt is {}'.format(filelist[-1]))
    return file_dir + filelist[-1]


def angle(vector1, vector2):
    cos_angle = vector1.dot(vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    angle = np.arccos(cos_angle)
    # angle2=angle*360/2/np.pi
    return angle  # , angle2


def area_triangle(vector1, vector2):
    trianglearea = 0.5 * np.linalg.norm( \
        np.cross(vector1, vector2))
    return trianglearea


def area_triangle_vertex(vertex1, vertex2, vertex3):
    trianglearea = 0.5 * np.linalg.norm( \
        np.cross(vertex2 - vertex1, vertex3 - vertex1))
    return trianglearea


def cal_angle_area(vector1, vector2):
    return angle(vector1, vector2), area_triangle(vector1, vector2)


# vij=np.array([ 0, 1,  1])
# vik=np.array([ 0, 2,  0])
# cal_angle_area(vij, vik)   # (0.7853981633974484, 1.0)


def cal_dist(vertex1, vertex2, ord=2):
    return np.linalg.norm(vertex1 - vertex2, ord=ord)
# vertex1 = np.array([1,2,3])
# vertex2 = np.array([4,5,6])
# cal_dist(vertex1, vertex2, ord=1), np.sum(vertex1-vertex2) # (9.0, -9)
# cal_dist(vertex1, vertex2, ord=2), np.sqrt(np.sum(np.square(vertex1-vertex2)))  # (5.196152422706632, 5.196152422706632)
# cal_dist(vertex1, vertex2, ord=3)  # 4.3267487109222245
