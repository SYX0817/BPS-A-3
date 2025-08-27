import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from dgl.nn.functional import edge_softmax

class MultiHeadAttentionSAGELayer(nn.Module):
    def __init__(self, ndim_in, edims, ndim_out, num_heads=4, dropout=0.1, use_layernorm=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = ndim_out // num_heads
        self.dropout = nn.Dropout(dropout)
        self.use_layernorm = use_layernorm

        input_dim = ndim_in + edims

        # Shared linear projection for both attention and message
        self.W_proj = nn.Linear(input_dim, ndim_out)  # SHARED

        # Learnable attention vector (like original GAT)
        self.attn_vec = nn.Parameter(th.FloatTensor(num_heads, self.head_dim))
        nn.init.xavier_uniform_(self.attn_vec)

        self.W_resid = nn.Linear(ndim_in, ndim_out) if ndim_in != ndim_out else nn.Identity()
        self.W_out = nn.Linear(ndim_out, ndim_out)

        if use_layernorm:
            self.norm = nn.LayerNorm(ndim_out)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, g, nfeats, efeats):
        with g.local_scope():
            g.ndata['h'] = nfeats
            g.edata['h_e'] = efeats

            def message_func(edges):
                x = th.cat([edges.src['h'], edges.data['h_e']], dim=1)  # (E, D_node + D_edge)
                x_proj = self.W_proj(x).view(-1, self.num_heads, self.head_dim)  # (E, H, D/H)

                # Compute attention scores: dot product with attn vector per head
                attn_scores = (x_proj * self.attn_vec).sum(dim=-1, keepdim=True)  # (E, H, 1)
                return {'e': attn_scores, 'm': x_proj}

            g.apply_edges(message_func)
            g.edata['alpha'] = edge_softmax(g, g.edata['e'])  # (E, H, 1)
            g.edata['m'] = self.dropout(g.edata['m'] * g.edata['alpha'])  # (E, H, D/H)

            g.update_all(fn.copy_e('m', 'm_sum'), fn.sum('m_sum', 'h_neigh'))  # (N, H, D/H)

            h_neigh = g.ndata['h_neigh'].reshape(-1, self.num_heads * self.head_dim)  # (N, D)
            h_self_proj = self.W_resid(nfeats)  # (N, D)
            h_out = F.relu(self.W_out(h_neigh + h_self_proj))  # Residual + Linear
            if self.use_layernorm:
                h_out = self.norm(h_out)
            return h_out

class EGAT(nn.Module):
    def __init__(self, ndim_in, ndim_out, edim, dropout, num_heads=4):
        super().__init__()
        self.layers = nn.ModuleList([
            MultiHeadAttentionSAGELayer(ndim_in, edim, 192, num_heads, dropout),
            MultiHeadAttentionSAGELayer(192, edim, ndim_out, num_heads, dropout)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, nfeats, efeats):
        for i, layer in enumerate(self.layers):
            if i != 0:
                nfeats = self.dropout(nfeats)
            nfeats = layer(g, nfeats, efeats)
        return nfeats

class MLPPredictor(nn.Module):
    def __init__(self, in_features, out_classes):
        super().__init__()
        self.W = nn.Linear(in_features * 2, out_classes)

    def apply_edges(self, edges):
        h_u = edges.src['h']
        h_v = edges.dst['h']
        score = self.W(th.cat([h_u, h_v], 1))
        return {'score': score}

    def forward(self, graph, h):
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(self.apply_edges)
            return graph.edata['score']

class DiEGAT(nn.Module):
    def __init__(self, ndim_in, ndim_out, edim, dropout, num_class, num_heads=2):
        super().__init__()
        self.gnn = EGAT(ndim_in, ndim_out, edim, dropout, num_heads)
        self.pred = MLPPredictor(ndim_out, num_class)

    def forward(self, g, nfeats, efeats):
        h = self.gnn(g, nfeats, efeats)
        return self.pred(g, h)
