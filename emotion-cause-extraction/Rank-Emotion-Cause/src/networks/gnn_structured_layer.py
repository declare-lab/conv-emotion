import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from config import DEVICE


class StructuredGraphAttentionLayer(nn.Module):
    def __init__(self, att_head, in_dim, out_dim, dp_gnn, leaky_alpha=0.2):
        super(StructuredGraphAttentionLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim*att_head
        self.dp_gnn = dp_gnn

        self.aggregator = nn.Linear(3*self.out_dim, self.out_dim)

        self.root_layer = nn.Linear(self.out_dim, 1)
        self.root_emb = nn.Parameter(torch.Tensor(1, self.in_dim))

        self.W = nn.Parameter(torch.Tensor(self.in_dim, self.out_dim))
        self.w_src = nn.Parameter(torch.Tensor(self.out_dim, 1))
        self.w_dst = nn.Parameter(torch.Tensor(self.out_dim, 1))
        self.leaky_alpha = leaky_alpha
        self.init_gnn_param()

        assert self.in_dim == self.out_dim
        self.H = nn.Linear(self.in_dim, self.in_dim)
        init.xavier_normal_(self.H.weight)

    def init_gnn_param(self):
        init.xavier_uniform_(self.aggregator.weight)
        init.xavier_uniform_(self.W.data)
        init.xavier_uniform_(self.w_src.data)
        init.xavier_uniform_(self.w_dst.data)

        init.xavier_uniform_(self.root_layer.weight)
        init.uniform_(self.root_emb.data, -0.1, 0.1)

    def forward(self, feat_in, adj):
        batch, N, in_dim = feat_in.size()
        assert in_dim == self.in_dim

        h = torch.matmul(feat_in, self.W)
        root_prob = self.root_layer(h).squeeze(2)

        attn_src = torch.matmul(F.tanh(h), self.w_src)
        attn_dst = torch.matmul(F.tanh(h), self.w_dst)
        attn = attn_src.expand(-1, -1, N) + attn_dst.expand(-1, -1, N).permute(0, 2, 1)
        attn = F.leaky_relu(attn, self.leaky_alpha, inplace=True)

        attn, root_prob = self.matrix_tree_theorem(attn, root_prob, adj)

        feat_c = torch.matmul(attn, feat_in)
        feat_p = torch.matmul(torch.transpose(attn, 1, 2), feat_in)
        feat_r = root_prob.unsqueeze(2).expand(-1, -1, self.in_dim) * self.root_emb.unsqueeze(0).expand(batch, N, -1)
        feat_out = torch.cat([feat_c, feat_p+feat_r, feat_in], 2)
        feat_out = self.aggregator(feat_out)
        feat_out = F.elu(feat_out)

        gate = F.sigmoid(self.H(feat_in))
        feat_out = gate * feat_out + (1 - gate) * feat_in
        feat_out = F.dropout(feat_out, self.dp_gnn, training=self.training)

        return feat_out, root_prob

    def matrix_tree_theorem(self, attn, root_prob, adj):
        """
        reference: https://github.com/nlpyang/SUMO/
        """
        adj = torch.FloatTensor(adj).to(DEVICE)
        mask = 1 - adj
        mask_ = mask[:, :, 0]
        root_prob = root_prob - mask_ * 50
        root_prob = torch.clamp(root_prob, min=-40)

        attn = attn - mask_.unsqueeze(1) * 50
        attn = attn - mask_.unsqueeze(2) * 50
        attn = torch.clamp(attn, min=-40)

        A = attn.exp()
        R = root_prob.exp()
        L = torch.sum(A, 1)
        L = torch.diag_embed(L)
        L = L - A

        LL = L + torch.diag_embed(R)
        LL_inv = torch.inverse(LL)
        LL_inv_diag = torch.diagonal(LL_inv, 0, 1, 2)

        root_prob = R * LL_inv_diag
        root_prob = root_prob.masked_fill(mask_.byte(), 0)

        LL_inv_diag = torch.unsqueeze(LL_inv_diag, 2)
        _A = torch.transpose(A, 1, 2)
        _A = _A * LL_inv_diag
        tmp1 = torch.transpose(_A, 1, 2)
        tmp2 = A * torch.transpose(LL_inv, 1, 2)
        attn = tmp1 - tmp2
        attn = torch.transpose(attn, 1, 2)
        attn = attn.masked_fill(mask.byte(), 0)

        return attn, root_prob

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_dim) + ' -> ' + str(self.out_dim) + ')'

