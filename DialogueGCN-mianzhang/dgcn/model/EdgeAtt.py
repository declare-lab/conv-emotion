import torch
import torch.nn as nn
import torch.nn.functional as F

import dgcn

log = dgcn.utils.get_logger()


class EdgeAtt(nn.Module):

    def __init__(self, g_dim, args):
        super(EdgeAtt, self).__init__()
        self.device = args.device
        self.wp = args.wp
        self.wf = args.wf

        self.weight = nn.Parameter(torch.zeros((g_dim, g_dim)).float(), requires_grad=True)
        var = 2. / (self.weight.size(0) + self.weight.size(1))
        self.weight.data.normal_(0, var)

    def forward(self, node_features, text_len_tensor, edge_ind):
        batch_size, mx_len = node_features.size(0), node_features.size(1)
        alphas = []

        weight = self.weight.unsqueeze(0).unsqueeze(0)
        att_matrix = torch.matmul(weight, node_features.unsqueeze(-1)).squeeze(-1)  # [B, L, D_g]
        for i in range(batch_size):
            cur_len = text_len_tensor[i].item()
            alpha = torch.zeros((mx_len, 110)).to(self.device)
            for j in range(cur_len):
                s = j - self.wp if j - self.wp >= 0 else 0
                e = j + self.wf if j + self.wf <= cur_len - 1 else cur_len - 1
                tmp = att_matrix[i, s: e + 1, :]  # [L', D_g]
                feat = node_features[i, j]  # [D_g]
                score = torch.matmul(tmp, feat)
                probs = F.softmax(score)  # [L']
                alpha[j, s: e + 1] = probs
            alphas.append(alpha)

        return alphas

# class EdgeAtt(nn.Module):
#
#     def __init__(self, g_dim, args):
#         super(EdgeAtt, self).__init__()
#         self.device = args.device
#         self.wp = args.wp
#         self.wf = args.wf
#         self.lin = nn.Linear(g_dim, 110)
#
#     def forward(self, node_features, text_len_tensor, edge_ind):
#         h = self.lin(node_features)  # [B, L, mx]
#         alphas = F.softmax(h, dim=-1)
#         # alphas = torch.ones((node_features.size(0), node_features.size(1), 110))
#         return alphas
