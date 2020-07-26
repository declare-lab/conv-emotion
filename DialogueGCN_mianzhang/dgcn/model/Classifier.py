import torch
import torch.nn as nn
import torch.nn.functional as F

import dgcn

log = dgcn.utils.get_logger()


class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_size, tag_size, args):
        super(Classifier, self).__init__()
        self.emotion_att = MaskedEmotionAtt(input_dim)
        self.lin1 = nn.Linear(input_dim, hidden_size)
        self.drop = nn.Dropout(args.drop_rate)
        self.lin2 = nn.Linear(hidden_size, tag_size)
        if args.class_weight:
            self.loss_weights = torch.tensor([1 / 0.086747, 1 / 0.144406, 1 / 0.227883,
                                              1 / 0.160585, 1 / 0.127711, 1 / 0.252668]).to(args.device)
            self.nll_loss = nn.NLLLoss(self.loss_weights)
        else:
            self.nll_loss = nn.NLLLoss()

    def get_prob(self, h, text_len_tensor):
        # h_hat = self.emotion_att(h, text_len_tensor)
        # hidden = self.drop(F.relu(self.lin1(h_hat)))
        hidden = self.drop(F.relu(self.lin1(h)))
        scores = self.lin2(hidden)
        log_prob = F.log_softmax(scores, dim=-1)

        return log_prob

    def forward(self, h, text_len_tensor):
        log_prob = self.get_prob(h, text_len_tensor)
        y_hat = torch.argmax(log_prob, dim=-1)

        return y_hat

    def get_loss(self, h, label_tensor, text_len_tensor):
        log_prob = self.get_prob(h, text_len_tensor)
        loss = self.nll_loss(log_prob, label_tensor)

        return loss


class MaskedEmotionAtt(nn.Module):

    def __init__(self, input_dim):
        super(MaskedEmotionAtt, self).__init__()
        self.lin = nn.Linear(input_dim, input_dim)

    def forward(self, h, text_len_tensor):
        batch_size = text_len_tensor.size(0)
        x = self.lin(h)  # [node_num, H]
        ret = torch.zeros_like(h)
        s = 0
        for bi in range(batch_size):
            cur_len = text_len_tensor[bi].item()
            y = x[s: s + cur_len]
            z = h[s: s + cur_len]
            scores = torch.mm(z, y.t())  # [L, L]
            probs = F.softmax(scores, dim=1)
            out = z.unsqueeze(0) * probs.unsqueeze(-1)  # [1, L, H] x [L, L, 1] --> [L, L, H]
            out = torch.sum(out, dim=1)  # [L, H]
            ret[s: s + cur_len, :] = out
            s += cur_len

        return ret


