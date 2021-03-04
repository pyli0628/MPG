import torch
import torch.nn as nn
from torch.nn import Linear, Sequential, GRU
import torch.nn.functional as F
from torch_geometric.nn import NNConv
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
from graph_bert import *

class CompoundProteinInteractionPrediction(nn.Module):
    def __init__(self,n_word, num_layer=5, dim=768, heads=12,  num_message_passing=3,
                 window=11,layer_cnn=3,layer_output=3,drop_ratio=0.5,graph_pooling='mean'):
        super().__init__()
        self.ligand_encoder = MolGT(num_layer, dim, heads, num_message_passing, drop_ratio)
        self.embed_word = nn.Embedding(n_word, dim)

        self.W_cnn = nn.ModuleList([nn.Conv2d(
                     in_channels=1, out_channels=1, kernel_size=2*window+1,
                     stride=1, padding=window) for _ in range(layer_cnn)])
        self.W_attention = nn.Linear(dim, dim)
        self.W_out = nn.ModuleList([nn.Linear(2*dim, 2*dim)
                                    for _ in range(layer_output)])
        self.W_interaction = nn.Linear(2*dim, 2)
        self.layer_cnn = layer_cnn
        self.layer_output = layer_output
        self.dummy=False
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            self.pool = GlobalAttention(gate_nn=torch.nn.Linear(emb_dim, 1))
        elif graph_pooling == "set2set":
            self.pool =nn.Sequential( Set2Set(emb_dim, 3), nn.Linear(2 * hidden_dim, hidden_dim) )
        elif graph_pooling == "dummy":
            self.dummy = True
        else:
            raise ValueError("Invalid graph pooling type.")

    def attention_cnn(self, x, xs, layer):
        """The attention mechanism is applied to the last layer of CNN."""
        # x: compound, xs: protein (n,len,hid)

        xs = torch.unsqueeze(xs, 1) # (n,1,len,hid)
        # print('xs',xs.shape)
        for i in range(layer):
            xs = torch.relu(self.W_cnn[i](xs))
        # print('xs1',xs.shape) #(n,1,len,hid)
        xs = torch.squeeze(xs, 1)
        # print('xs2',xs.shape)# (n,len,hid)

        h = torch.relu(self.W_attention(x)) #n,hid
        hs = torch.relu(self.W_attention(xs))#n,len,hid
        weights =  torch.tanh(torch.bmm(h.unsqueeze(1),hs.permute(0,2,1))) #torch.tanh(F.linear(h, hs))#n,len
        ys = weights.permute(0,2,1) * hs #n,l,h
        # return torch.unsqueeze(torch.sum(ys, 0), 0)
        return torch.mean(ys, 1)

    def forward(self, inputs):

        ligand, words = inputs,inputs.protein

        """Compound vector with GNN."""

        compound_vector = self.ligand_encoder.forward(ligand)
        if self.dummy:
            compound_vector =compound_vector[ligand.ummy_indice]
        else:
            compound_vector = self.pool(compound_vector, ligand.batch)
        # print('coumpound',compound_vector.shape) #(1,hid)

        """Protein vector with attention-CNN."""
        word_vectors = self.embed_word(words)
        # print('word',word_vectors.shape) #(len,hid)

        protein_vector = self.attention_cnn(compound_vector,
                                            word_vectors, self.layer_cnn)
        # print('protein',[protein_vector.shape]) #(1,hid)

        """Concatenate the above two vectors and output the interaction."""
        cat_vector = torch.cat((compound_vector, protein_vector), 1)
        for j in range(self.layer_output):
            cat_vector = torch.relu(self.W_out[j](cat_vector))
        interaction = self.W_interaction(cat_vector)

        return interaction
    def from_pretrain(self,gnn_file):
        self.ligand_encoder.load_state_dict(torch.load(gnn_file,map_location=torch.device('cpu')))


class CompoundProteinInteractionPrediction_old(nn.Module):
    def __init__(self,n_word, num_layer=5, dim=768, heads=12,  num_message_passing=3,
                 window=11,layer_cnn=3,layer_output=3,drop_ratio=0.5,graph_pooling='mean'):
        super().__init__()
        self.ligand_encoder = MolGT(num_layer, dim, heads, num_message_passing, drop_ratio)
        self.embed_word = nn.Embedding(n_word, dim)

        self.W_cnn = nn.ModuleList([nn.Conv2d(
                     in_channels=1, out_channels=1, kernel_size=2*window+1,
                     stride=1, padding=window) for _ in range(layer_cnn)])
        self.W_attention = nn.Linear(dim, dim)
        self.W_out = nn.ModuleList([nn.Linear(2*dim, 2*dim)
                                    for _ in range(layer_output)])
        self.W_interaction = nn.Linear(2*dim, 2)
        self.layer_cnn = layer_cnn
        self.layer_output = layer_output

        self.dummy=False
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            self.pool = GlobalAttention(gate_nn=torch.nn.Linear(emb_dim, 1))
        elif graph_pooling == "set2set":
            self.pool =nn.Sequential( Set2Set(emb_dim, 3), nn.Linear(2 * hidden_dim, hidden_dim) )
        elif graph_pooling == "dummy":
            self.dummy = True
        else:
            raise ValueError("Invalid graph pooling type.")


    def attention_cnn(self, x, xs, layer):
        """The attention mechanism is applied to the last layer of CNN."""

        xs = torch.unsqueeze(torch.unsqueeze(xs, 0), 0)
        for i in range(layer):
            xs = torch.relu(self.W_cnn[i](xs))
        xs = torch.squeeze(torch.squeeze(xs, 0), 0)

        h = torch.relu(self.W_attention(x))
        hs = torch.relu(self.W_attention(xs))
        weights = torch.tanh(F.linear(h, hs))
        ys = torch.t(weights) * hs

        # return torch.unsqueeze(torch.sum(ys, 0), 0)
        return torch.unsqueeze(torch.mean(ys, 0), 0)

    def forward(self, inputs):

        ligand, words = inputs,inputs.protein

        """Compound vector with GNN."""

        compound_vector = self.ligand_encoder.forward(ligand)
        if self.dummy:
            compound_vector =compound_vector[ligand.ummy_indice]
        else:
            compound_vector = self.pool(compound_vector, ligand.batch)

        """Protein vector with attention-CNN."""
        word_vectors = self.embed_word(words)
        protein_vector = self.attention_cnn(compound_vector,
                                            word_vectors, self.layer_cnn)

        """Concatenate the above two vectors and output the interaction."""
        cat_vector = torch.cat((compound_vector, protein_vector), 1)
        for j in range(self.layer_output):
            cat_vector = torch.relu(self.W_out[j](cat_vector))
        interaction = self.W_interaction(cat_vector)

        return interaction
    def from_pretrain(self,gnn_file):
        self.ligand_encoder.load_state_dict(torch.load(gnn_file,map_location=torch.device('cpu'))['gnn'])




