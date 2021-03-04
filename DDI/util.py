import torch
import random
from loader import *





# add by dd
from torch_geometric.utils import to_dense_adj
from torch_geometric.utils import add_self_loops
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
        num_nodes = data.num_nodes
        num_edges = data.num_edges
        node_seg = [0 for _ in range(num_nodes)]
        edge_seg = [0 for _ in range(num_edges)]
        data.edge_seg = torch.LongTensor(edge_seg)
        data.node_seg = torch.LongTensor(node_seg)
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


class Random_graph:
    def __init__(self):
        self.add_loop = Self_loop()
    def __call__(self, data, data_random=None):
        data = self.add_loop(data)
        data_random = self.add_loop(data_random)
        data = self.random_graph(data,data_random)
        return data
    def disconnect_self(self,data):
        num_atoms = data.x.size()[0]
        # adj = to_dense_adj(data.edge_index).squeeze()
        edge_index = data.edge_index.t()
        edge_attr = data.edge_attr
        # disconnect mol from atom n
        disconnect_edge_index = []
        patience = 0
        while len(disconnect_edge_index) < 1:
            sample_idx = []
            edge_seg = []
            n = random.randint(int(num_atoms / 3), int(num_atoms* 2 / 3))
            node_seg = [0 for _ in range(n)] + [1 for _ in range(num_atoms - n)]
            for i, item in enumerate(edge_index):
                if item[0] < n and item[1] < n:
                    sample_idx.append(i)
                    edge_seg.append(0)
                elif item[0] >= n and item[1] >= n:
                    sample_idx.append(i)
                    edge_seg.append(1)
                else:
                    continue
            disconnect_edge_index = edge_index[torch.LongTensor(sample_idx)]
            disconnect_edge_attr=edge_attr[torch.LongTensor(sample_idx)]
            patience += 1
            if patience > 500:
                print('Note: cant not disconnect this molecule')
                data.node_seg = torch.LongTensor(node_seg)
                data.edge_seg = torch.LongTensor(edge_seg)
                data.ngp_y = torch.LongTensor([1])
                return data
            # print('self disc',n)
        data.edge_index = disconnect_edge_index.t()
        data.edge_attr = disconnect_edge_attr
        data.edge_seg = torch.LongTensor(edge_seg)
        data.node_seg = torch.LongTensor(node_seg)
        data.ngp_y = torch.LongTensor([1])
        return data

    def random_graph(self,data,data_random):
        if random.random() > 0.5:
            return self.disconnect_self(data)
        else:
            num_atoms = data.x.size()[0]
            # adj = to_dense_adj(data.edge_index).squeeze()
            edge_index = data.edge_index.t()
            edge_attr = data.edge_attr
            num_atoms_rd = data_random.x.size()[0]
            edge_index_rd = data_random.edge_index.t()
            edge_attr_rd = data_random.edge_attr
            if num_atoms<3:
                # print('Note: Atom num less than 3')
                node_seg = [0 for _ in range(num_atoms)]
                edge_seg = [0 for _ in range(len(edge_attr))]
                data.edge_seg = torch.LongTensor(edge_seg)
                data.node_seg = torch.LongTensor(node_seg)
                data.ngp_y = torch.LongTensor([1])
                return data
            elif num_atoms_rd<3:
                node_seg = [0 for _ in range(num_atoms)]+[1 for _ in range(num_atoms_rd)]
                edge_seg = [0 for _ in range(len(edge_attr))]+[1 for _ in range(len(edge_attr_rd))]
                data.x = torch.cat([data.x, data_random.x],dim=0)
                data.edge_index = torch.cat([data.edge_index,data_random.edge_index],dim=1)
                data.edge_attr = torch.cat([data.edge_attr,data_random.edge_attr],dim=0)
                data.edge_seg = torch.LongTensor(edge_seg)
                data.node_seg = torch.LongTensor(node_seg)
                data.ngp_y = torch.LongTensor([0])
                return data

            # disconnect mol from atom n
            disconnect_edge_index = []
            disconnect_edge_attr = []
            edge_seg = []
            node_seg = []
            patience = 0
            while len(disconnect_edge_index)<1:
                if random.random() > 0.5:#upper-upper
                    sample_idx = []

                    n = random.randint(int(num_atoms / 3), int(num_atoms * 2 / 3))

                    node_seg.extend([0 for _ in range(n)])
                    for i, item in enumerate(edge_index):
                        if item[0] < n and item[1] < n:
                            sample_idx.append(i)
                            edge_seg.append(0)
                    edge_index_a = edge_index[torch.LongTensor(sample_idx)]
                    edge_attr_a = edge_attr[torch.LongTensor(sample_idx)]


                    sample_idx = []
                    n_rd = random.randint(int(num_atoms_rd / 3), int(num_atoms_rd * 2 / 3))
                    node_seg.extend([1 for _ in range(n_rd)])
                    for i, item in enumerate(edge_index_rd):
                        if item[0] < n_rd and item[1] < n_rd:
                            sample_idx.append(i)
                            edge_seg.append(1)
                    edge_index_b = edge_index_rd[torch.LongTensor(sample_idx)]+n
                    edge_attr_b = edge_attr_rd[torch.LongTensor(sample_idx)]


                    disconnect_edge_index=torch.cat([edge_index_a,edge_index_b],dim=0)
                    disconnect_edge_attr=torch.cat([edge_attr_a,edge_attr_b],dim=0)
                    disconnect_node = torch.cat([data.x[:n,:],data_random.x[:n_rd,:]],dim=0)
                    assert disconnect_edge_index.max()<len(disconnect_node)
                    # print('up-up',n, n_rd)
                else: #upper-lower
                    sample_idx = []
                    n = random.randint(int(num_atoms/ 3), int(num_atoms * 2 / 3))
                    node_seg.extend([0 for _ in range(n)])
                    for i, item in enumerate(edge_index):
                        if item[0] < n and item[1] < n:
                            sample_idx.append(i)
                            edge_seg.append(0)
                    edge_index_a = edge_index[torch.LongTensor(sample_idx)]
                    edge_attr_a = edge_attr[torch.LongTensor(sample_idx)]

                    sample_idx = []
                    n_rd = random.randint(int(num_atoms_rd / 3), int(num_atoms_rd * 2 / 3))
                    node_seg.extend([1 for _ in range(num_atoms_rd-n_rd)])
                    for i, item in enumerate(edge_index_rd):
                        if item[0] >= n_rd and item[1] >= n_rd:
                            sample_idx.append(i)
                            edge_seg.append(1)
                    edge_index_b = edge_index_rd[torch.LongTensor(sample_idx)] -n_rd + n
                    edge_attr_b = edge_attr_rd[torch.LongTensor(sample_idx)]

                    disconnect_edge_index=torch.cat([edge_index_a,edge_index_b],dim=0)
                    disconnect_edge_attr=torch.cat([edge_attr_a,edge_attr_b],dim=0)
                    disconnect_node = torch.cat([data.x[:n,:],data_random.x[n_rd:,:]],dim=0)
                    assert disconnect_edge_index.max()<len(disconnect_node)
                    # print('up-low',n, n_rd)
                patience+=1
                if patience>500:
                    print('Note: cant not random combine this molecule!')
                    return disconnect_self(data)
            assert len(disconnect_edge_attr)==len(disconnect_edge_index)
            data.x = torch.LongTensor(disconnect_node)
            data.edge_index = torch.LongTensor(disconnect_edge_index).t()
            data.edge_attr = torch.LongTensor(disconnect_edge_attr)
            data.edge_seg = torch.LongTensor(edge_seg)
            data.node_seg = torch.LongTensor(node_seg)
            data.ngp_y = torch.LongTensor([0])
            return data



class MaskAtom:
    def __init__(self, num_atom_type, num_edge_type, mask_rate, mask_edge=True):
        """
        Randomly masks an atom, and optionally masks edges connecting to it.
        The mask atom type index is num_possible_atom_type
        The mask edge type index in num_possible_edge_type
        :param num_atom_type:
        :param num_edge_type:
        :param mask_rate: % of atoms to be masked
        :param mask_edge: If True, also mask the edges that connect to the
        masked atoms
        """
        self.num_atom_type = num_atom_type
        self.num_edge_type = num_edge_type
        self.mask_rate = mask_rate
        self.mask_edge = mask_edge


    def __call__(self, data, data_random=None, masked_atom_indices=None):

        if masked_atom_indices == None:
            # sample x distinct atoms to be masked, based on mask rate. But
            # will sample at least 1 atom
            num_atoms = data.x.size()[0]
            sample_size = int(num_atoms * self.mask_rate + 1)
            masked_atom_indices = random.sample(range(num_atoms), sample_size)

        data.masked_atom_indices = torch.tensor(masked_atom_indices)
        data.mask_node_label = data.x[data.masked_atom_indices][:, 0].clone()
        data.x[data.masked_atom_indices] = torch.tensor([[119,121,133,138,146,156,164,176]])

        return data

    def __repr__(self):
        return '{}(num_atom_type={}, num_edge_type={}, mask_rate={}, mask_edge={})'.format(
            self.__class__.__name__, self.num_atom_type, self.num_edge_type,
            self.mask_rate, self.mask_edge)
class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (list of :obj:`transform` objects): List of transforms to
            compose.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data,data_random=None):
        for t in self.transforms:
            data = t(data,data_random)
        return data

    def __repr__(self):
        args = ['    {},'.format(t) for t in self.transforms]
        return '{}([\n{}\n])'.format(self.__class__.__name__, '\n'.join(args))
