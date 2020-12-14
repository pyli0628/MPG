import torch,math
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.nn.inits import glorot, zeros
from torch.nn import Parameter
from torch import nn

num_atom_type = 120+1+1 #including the extra mask tokens+master_node+bridge_node
num_chirality_tag = 3

num_bond_type = 6+1+1 #including aromatic and self-loop edge, and extra masked tokens+master_node+bridge
num_bond_direction = 3

seg_size = 3

try:
    import apex
    #apex.amp.register_half_function(apex.normalization.fused_layer_norm, 'FusedLayerNorm')
    import apex.normalization
    from apex.normalization.fused_layer_norm import FusedLayerNormAffineFunction
    #apex.amp.register_float_function(apex.normalization.FusedLayerNorm, 'forward')
    #BertLayerNorm = apex.normalization.FusedLayerNorm
    APEX_IS_AVAILABLE = True
except ImportError:
    print("Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex.")
    #BertLayerNorm = BertNonFusedLayerNorm
    APEX_IS_AVAILABLE = False
class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super(BertLayerNorm, self).__init__()
        self.shape = torch.Size((hidden_size,))
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.apex_enabled = APEX_IS_AVAILABLE

    @torch.jit.unused
    def fused_layer_norm(self, x):
        return FusedLayerNormAffineFunction.apply(
                    x, self.weight, self.bias, self.shape, self.eps)


    def forward(self, x):
        if self.apex_enabled and not torch.jit.is_scripting():
            x = self.fused_layer_norm(x)
        else:
            u = x.mean(-1, keepdim=True)
            s = (x - u).pow(2).mean(-1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight * x + self.bias
        return x

class AttentionOut(nn.Module):
    def __init__(self, hidden,dropout):
        super(AttentionOut, self).__init__()
        self.dense = nn.Linear(hidden, hidden)
        self.LayerNorm = BertLayerNorm(hidden, eps=1e-12)
        self.dropout = nn.Dropout(dropout)
    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / 1.41421))
def bias_gelu(bias, y):
    x = bias + y
    return x * 0.5 * (1.0 + torch.erf(x / 1.41421))
class LinearActivation(nn.Module):
    r"""Fused Linear and activation Module.
    """
    def __init__(self, in_features, out_features,  bias=True):
        super(LinearActivation, self).__init__()
        self.in_features = in_features
        self.out_features = out_features                                                                 #
        if bias:  # compatibility
            self.biased_act_fn =bias_gelu
        else:
            self.act_fn = gelu
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        if not self.bias is None:
            return self.biased_act_fn(self.bias, F.linear(input, self.weight, None))
        else:
            return self.act_fn(F.linear(input, self.weight, self.bias))

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

class Intermediate(nn.Module):
    def __init__(self, hidden):
        super(Intermediate, self).__init__()
        self.dense_act = LinearActivation(hidden, 4*hidden)
    def forward(self, hidden_states):
        hidden_states = self.dense_act(hidden_states)
        return hidden_states

class GTOut(nn.Module):
    def __init__(self, hidden,dropout):
        super(GTOut, self).__init__()
        self.dense = nn.Linear(hidden*4, hidden)
        self.LayerNorm = BertLayerNorm(hidden, eps=1e-12)
        self.dropout = nn.Dropout(dropout)
    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class GraphAttentionConv(MessagePassing):
    def __init__(self, hidden, heads=3,dropout=0.):
        super(GraphAttentionConv, self).__init__()  # aggr='mean'
        self.hidden=hidden
        self.heads = heads
        assert hidden%heads==0
        self.query = nn.Linear(hidden, heads * int(hidden/heads))
        self.key = nn.Linear(hidden, heads * int(hidden/heads))
        self.value = nn.Linear(hidden, heads * int(hidden/heads))
        self.attn_drop = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.query.weight.data)
        torch.nn.init.xavier_uniform_(self.key.weight.data)
        torch.nn.init.xavier_uniform_(self.value.weight.data)

    def forward(self, x, edge_index, edge_attr, size=None):
        pseudo = edge_attr.unsqueeze(-1) if edge_attr.dim() == 1 else edge_attr
        return self.propagate(edge_index, size=size, x=x, pseudo=pseudo)

    def message(self, edge_index_i, x_i, x_j, pseudo, size_i):
        query =self.query(x_i).view(-1, self.heads, int(self.hidden/self.heads))
        key = self.key(x_j+pseudo).view(-1, self.heads, int(self.hidden/self.heads))
        value = self.value(x_j+pseudo).view(-1, self.heads, int(self.hidden/self.heads))

        alpha = (query * key).sum(dim=-1)/math.sqrt(int(self.hidden/self.heads))
        alpha = softmax(alpha, edge_index_i, size_i)
        alpha =self.attn_drop(alpha.view(-1, self.heads, 1))
        return alpha * value
    def update(self, aggr_out):
        aggr_out = aggr_out.view(-1, self.heads * int(self.hidden/self.heads))
        return aggr_out

class GTLayer(nn.Module):
    def __init__(self, hidden, heads,  dropout,num_message_passing):
        super(GTLayer, self).__init__()
        self.attention = GraphAttentionConv(hidden, heads, dropout)
        self.att_out = AttentionOut(hidden,dropout)
        self.intermediate = Intermediate(hidden)
        self.output = GTOut(hidden,dropout)
        self.gru = nn.GRU(hidden, hidden)
        self.LayerNorm = BertLayerNorm(hidden, eps=1e-12)
        self.time_step = num_message_passing
    def forward(self, x, edge_index, edge_attr):
        h = x.unsqueeze(0)
        for i in range(self.time_step):
            attention_output = self.attention.forward(x, edge_index, edge_attr)
            attention_output = self.att_out.forward(attention_output,x)
            intermediate_output = self.intermediate.forward(attention_output)
            m = self.output.forward(intermediate_output, attention_output)
            x, h = self.gru(m.unsqueeze(0), h)
            x = self.LayerNorm.forward(x.squeeze(0))
        return x

class MolGNet(torch.nn.Module):

    def __init__(self, num_layer, emb_dim, heads, num_message_passing, drop_ratio = 0):
        super(MolGNet, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.x_embedding = torch.nn.Embedding(178, emb_dim)
        # self.x_embedding2 = torch.nn.Embedding(num_chirality_tag, emb_dim)
        self.x_seg_embed = torch.nn.Embedding(seg_size,emb_dim)

        self.edge_embedding = torch.nn.Embedding(18, emb_dim)
        # self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)
        self.edge_seg_embed = torch.nn.Embedding(seg_size,emb_dim)

        self.reset_parameters()

        self.gnns = torch.nn.ModuleList(
            [GTLayer(emb_dim, heads, drop_ratio, num_message_passing) for _ in range(num_layer)])

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.x_embedding.weight.data)

        torch.nn.init.xavier_uniform_(self.x_seg_embed.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding.weight.data)

        torch.nn.init.xavier_uniform_(self.edge_seg_embed.weight.data)
    #def forward(self, x, edge_index, edge_attr):
    def forward(self, *argv):
        if len(argv) == 5:
            x, edge_index, edge_attr,node_seg,edge_seg = argv[0], argv[1], argv[2],argv[3],argv[4]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr,node_seg,edge_seg = data.x, data.edge_index, data.edge_attr,data.node_seg,data.edge_seg
        else:
            raise ValueError("unmatched number of arguments.")

        x = self.x_embedding(x).sum(1) + self.x_seg_embed(node_seg)

        edge_attr = self.edge_embedding(edge_attr).sum(1) + self.edge_seg_embed(edge_seg)
        for gnn in self.gnns:
            x = gnn(x,edge_index,edge_attr)
        return x

class Pretrain_MolGNet(torch.nn.Module):
    def __init__(self, num_layer, emb_dim, heads, num_message_passing, drop_ratio=0):
        super(Pretrain_MolGNet, self).__init__()
        self.emb_dim = emb_dim
        self.gnn = MolGNet(num_layer, emb_dim, heads, num_message_passing, drop_ratio)
        self.linear_pred_atoms = torch.nn.Linear(emb_dim, 119)
        self.linear_pred_cgp = torch.nn.Linear(emb_dim, 2)
    def forward(self,data):
        node_rep = self.gnn.forward(data)
        pred_node = self.linear_pred_atoms(node_rep[data.masked_atom_indices])
        graph_rep = self.linear_pred_cgp(node_rep[data.dummy_node_indices])
        return pred_node,graph_rep

class MolGT_graphpred(torch.nn.Module):

    def __init__(self, num_layer, emb_dim, heads, num_message_passing, num_tasks, drop_ratio=0, graph_pooling="mean"):
        super(MolGT_graphpred, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.gnn = MolGT(num_layer, emb_dim, heads, num_message_passing, drop_ratio)
        self.dummy=False
        # Different kind of graph pooling
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            self.pool = GlobalAttention(gate_nn=torch.nn.Linear(emb_dim, 1))
        elif graph_pooling == "set2set":
            self.pool = Set2Set(emb_dim, 3)
        elif graph_pooling == "dummy":
            self.dummy = True
        else:
            raise ValueError("Invalid graph pooling type.")

        # For graph-level binary classification
        if graph_pooling == "set2set":
            self.mult = 2
        else:
            self.mult = 1

        self.graph_pred_linear = torch.nn.Linear(self.mult * self.emb_dim, self.num_tasks)
    def from_pretrained(self, model_file):
        # self.gnn = GNN(self.num_layer, self.emb_dim, JK = self.JK, drop_ratio = self.drop_ratio)
        self.gnn.load_state_dict(torch.load(model_file)["gnn"])

    def forward(self, *argv):
        if len(argv) == 4:
            x, edge_index, edge_attr, batch,node_seg,edge_seg,dummy_indice =\
                argv[0], argv[1], argv[2], argv[3], argv[4], argv[5],argv[6]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr, batch, node_seg,edge_seg,dummy_indice = \
                data.x, data.edge_index, data.edge_attr, data.batch,data.node_seg,data.edge_seg,data.dummy_node_indices
        else:
            raise ValueError("unmatched number of arguments.")

        node_representation = self.gnn(x, edge_index, edge_attr,node_seg,edge_seg)
        if self.dummy:
            return self.graph_pred_linear(node_representation[dummy_indice])
        else:
            return self.graph_pred_linear(self.pool(node_representation, batch))


if __name__ == "__main__":
    pass

