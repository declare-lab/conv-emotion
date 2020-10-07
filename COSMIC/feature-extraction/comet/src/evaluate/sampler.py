import torch
import torch.nn as nn
import torch.nn.functional as F

import comet.src.data.data as data
import comet.src.data.config as cfg
import comet.src.models.utils as model_utils
import comet.src.evaluate.utils as eval_utils
import comet.src.train.batch as batch_utils

def make_sampler(sampler_type, opt, *args, **kwargs):
    print("Initializing Greedy Sampler")
    return GreedySampler(opt, *args, **kwargs)

class Sampler():
    def __init__(self, opt, data_loader, batch_mode=False):
        # Token on which to end sampling
        self.end_token = data_loader.vocab_encoder[data.end_token]

        self.opt = opt

    def generate_sequence(self, batch, model):
        raise


class GreedySampler(Sampler):
    def __init__(self, opt, data_loader, batch_mode=True):
        super(GreedySampler, self).__init__(opt, data_loader)

    def append_batch(self, X, next_idx, mask):
        next_pos = X[:, -1:, 1] + 1
        next_x = torch.cat((next_idx, next_pos), -1).unsqueeze(1)
        next_mask = torch.cat([mask, torch.ones(X.size(0), 1, device=mask.device)], 1)
        return torch.cat((X, next_x), 1), next_mask

    def generate_sequence(self, batch, model, data_loader, start_idx, end_len):
        XMB = batch["sequences"][:, :start_idx]
        MMB = batch["attention_mask"][:, :start_idx]

        XMB = model_utils.prepare_position_embeddings(
            self.opt, data_loader.vocab_encoder, XMB.unsqueeze(-1))

        _, lp = model(
            XMB.unsqueeze(1), sequence_mask=MMB)
        lm_probs = F.log_softmax(lp, dim=-1)

        values, indices = lm_probs[:, -1, :].max(dim=-1)
        seqs = indices.clone().unsqueeze(1)

        loss = values
        counts = 1
        next_pos = XMB[:, -1:, 1] + 1
        next_x = torch.cat((indices.view(-1, 1), next_pos), -1).unsqueeze(1)
        XMB = torch.cat((XMB, next_x), 1)
        MMB = torch.cat([MMB, torch.ones(XMB.size(0), 1, device=MMB.device)], 1)

        # Sample from top k

        for _ in range(self.opt.eval.smax):
            _, lp = model(
                XMB.unsqueeze(1), sequence_mask=MMB)
            lm_probs = F.log_softmax(lp, dim=-1)

            # Sample from top k
            values, next_idx = lm_probs[:, -1, :].max(dim=-1)

            loss += values
            counts += 1

            next_idx = next_idx.unsqueeze(1)

            seqs = torch.cat([seqs, next_idx], 1)

            if (next_idx.item() == self.end_token) or (_ == end_len - 1):
                break

            XMB, MMB = self.append_batch(XMB, next_idx, MMB)

        beams = []

        for beam in seqs:
            beams.append(" ".join("".join(
                [data_loader.vocab_decoder[tok.item()].replace(
                    '</w>', ' ').replace('\n', '')
                 for tok in beam if tok != self.end_token]).split()))

        sampling_result = {
            "sequence": beams[0],
            "beams": beams,
            "beam_losses": [loss.item()],
            "loss": loss.item(),
            "beam_lengths": [counts],
            "length": counts
        }

        return sampling_result


class TopKSampler(Sampler):
    def __init__(self, opt, data_loader, batch_mode=True):
        super(TopKSampler, self).__init__(opt, data_loader)

    def append_batch(self, X, next_idx, mask):
        next_pos = X[:, -1:, 1] + 1
        next_x = torch.cat((next_idx, next_pos), -1).unsqueeze(1)
        next_mask = torch.cat([mask, torch.ones(X.size(0), 1, device=mask.device)], 1)
        return torch.cat((X, next_x), 1), next_mask

    def generate_sequence(self, batch, model, data_loader, start_idx, end_len):
        # start_idx = context_size_event + 1
        # start_idx = max_e1 + max_r
        # end_idx = context_size_effect - 1
        # end_idx = max_e2
        XMB = batch["sequences"][:, :start_idx]
        MMB = batch["attention_mask"][:, :start_idx]

        XMB = model_utils.prepare_position_embeddings(
            self.opt, data_loader.vocab_encoder, XMB.unsqueeze(-1))

        _, lp = model(
            XMB.unsqueeze(1), sequence_mask=MMB)
        lm_probs = F.log_softmax(lp, dim=-1)

        values, indices = lm_probs[:, -1, :].topk(self.opt.eval.k)
        seqs = indices.t().clone()

        losses = - values.view(-1, 1)

        ended = (seqs == self.end_token).float()
        counts = (1 - ended)
        XMB = XMB.repeat(self.opt.eval.k, 1, 1)
        MMB = MMB.repeat(self.opt.eval.k, 1)
        next_pos = XMB[:, -1:, 1] + 1
        next_x = torch.cat((indices.view(self.opt.eval.k, -1), next_pos), -1).unsqueeze(1)
        XMB = torch.cat((XMB, next_x), 1)
        MMB = torch.cat([MMB, torch.ones(XMB.size(0), 1, device=MMB.device)], 1)

        # Sample from top k

        for _ in range(end_len):
            _, lp = model(XMB.unsqueeze(1), sequence_mask=MMB)
            lm_probs = F.log_softmax(lp, dim=-1)

            # Sample from top k
            values, indices = lm_probs[:, -1, :].topk(self.opt.eval.k)
            choice = torch.multinomial(values.exp(), 1)
            next_idx = indices.gather(-1, choice)

            ended = ended + (next_idx == self.end_token).float() * (1 - ended)

            next_idx = next_idx * (1 - ended).long() + ended.long() * self.end_token

            counts += (1 - ended)

            seqs = torch.cat([seqs, next_idx], 1)

            if ended.sum().item() == self.opt.eval.k:
                break

            losses -= values.gather(-1, choice) * (1 - ended)

            XMB, MMB = self.append_batch(XMB, next_idx, MMB)

        beams = []

        for beam in seqs:
            beams.append(" ".join("".join(
                [data_loader.vocab_decoder[tok.item()].replace(
                    '</w>', ' ').replace('\n', '')
                 for tok in beam if tok != self.end_token]).split()))

        sampling_result = {
            "sequence": beams[0],
            "beams": beams,
            "beam_losses": losses.squeeze().tolist(),
            "loss": losses[0].item(),
            "beam_lengths": counts.long().squeeze().tolist(),
            "length": counts[0].long().item()
        }

        return sampling_result


class BeamSampler(TopKSampler):
    def __init__(self, opt, data_loader, batch_mode=True, scorer=None):
        super(BeamSampler, self).__init__(opt, data_loader, batch_mode)

        self.kill_mask = torch.ones(opt.eval.bs, opt.eval.bs).to(cfg.device) * 9000
        self.kill_mask[:, 0] = 0

    def make_batch(self, X):
        X = np.array(X)
        assert X.ndim in [1, 2]
        if X.ndim == 1:
            X = np.expand_dims(X, axis=0)
        pos_enc = np.arange(n_vocab + n_special, n_vocab + n_special + X.shape[-1])
        pos_enc = np.expand_dims(pos_enc, axis=0)
        batch = np.stack([X, pos_enc], axis=-1)
        batch = torch.tensor(batch, dtype=torch.long).to(device)
        return batch

    def append_batch(self, X, beam_toks, mask):
        next_pos = X[:, -1:, 1] + 1
        next_x = torch.cat((beam_toks.unsqueeze(1), next_pos), -1).unsqueeze(1)
        next_mask = torch.cat([mask, torch.ones(X.size(0), 1, device=mask.device)], 1)
        return torch.cat((X, next_x), 1), next_mask

    def generate_sequence(self, batch, model, data_loader, start_idx, end_len):
        # start_idx = context_size_event + 1
        # start_idx = max_e1 + max_r
        # end_idx = context_size_effect - 1
        # end_idx = max_e2
        XMB = batch["sequences"][:, :start_idx]
        MMB = batch["attention_mask"][:, :start_idx]

        XMB = model_utils.prepare_position_embeddings(
            self.opt, data_loader.vocab_encoder, XMB.unsqueeze(-1))

        tokens = []
        beam_losses = []
        # Beam Search
        beam_lls, beam_toks, beam_seqs = None, None, None
        _, lp = model(XMB.unsqueeze(1), sequence_mask=MMB)
        lm_probs = F.log_softmax(lp, dim=-1)
        dist = lm_probs[:, -1, :].squeeze()
        beam_lls, beam_toks = dist.topk(self.opt.eval.bs)
        beam_losses.append(beam_lls)

        ended = (beam_toks == self.end_token).float()
        counts = (2 - ended)
        beam_toks = beam_toks.unsqueeze(1)
        beam_seqs = beam_toks.clone()
        XMB = XMB.repeat(self.opt.eval.bs, 1, 1)
        MMB = MMB.repeat(self.opt.eval.bs, 1)
        next_pos = XMB[:, -1:, 1] + 1
        next_x = torch.cat((beam_toks, next_pos), -1).unsqueeze(1)
        XMB = torch.cat((XMB, next_x), 1)
        MMB = torch.cat([MMB, torch.ones(XMB.size(0), 1, device=MMB.device)], 1)

        for _ in range(end_len):

            # Compute distribution for current beam
            _, lp = model(
                XMB.unsqueeze(1), sequence_mask=MMB)
            lm_probs = F.log_softmax(lp, dim=-1)
            dist = lm_probs[:, -1, :].squeeze()

            # get hypothesis tokens for distribution
            hyp_beam_lls, hyp_beam_toks = dist.topk(self.opt.eval.bs)

            # Compute masks and expand beam
            expanded_ended = ended.unsqueeze(1).repeat(1, self.opt.eval.bs)
            hypothesis_mask = expanded_ended * self.kill_mask + (1 - expanded_ended)

            paper_results = False

            if paper_results:
                # Results from paper with slightly buggy beam search
                current_beam_lls = beam_lls.unsqueeze(1).repeat(
                    1, self.opt.eval.bs).view(self.opt.eval.bs**2)
            else:
                # Current beam search implementation
                current_beam_lls = beam_losses[-1].unsqueeze(1).repeat(
                    1, self.opt.eval.bs).view(self.opt.eval.bs**2)

            # Compute losses of hypotheses, masking those that have ended
            hyp_beam_lls = (hyp_beam_lls.view(self.opt.eval.bs**2) *
                            hypothesis_mask.view(-1)) + current_beam_lls

            # Get normalizer for sequences
            temp_counts = counts.unsqueeze(1).repeat(1, self.opt.eval.bs).view(
                self.opt.eval.bs ** 2)

            # Select best beams with lowest aggregate loss
            beam_lls, top_beam_idxs = (hyp_beam_lls / temp_counts).topk(self.opt.eval.bs)

            # Update placements in beam based on selecetion
            beam_losses = [i.index_select(0, top_beam_idxs // self.opt.eval.bs)
                           for i in beam_losses]
            ended = ended.index_select(0, top_beam_idxs // self.opt.eval.bs)
            counts = temp_counts.index_select(0, top_beam_idxs)

            # Save beam losses
            beam_losses.append(beam_lls * counts)

            # Update beam tokens
            ended_mask = (1 - ended).long()
            end_replacement = (self.end_token * ended).long()
            next_toks = hyp_beam_toks.view(-1)[top_beam_idxs]
            beam_toks = next_toks * ended_mask + end_replacement

            # Update ended and counts
            ended = ended + (beam_toks == self.end_token).float() * (1 - ended)
            counts = counts + (1 - ended)

            # Update beam sequences
            beam_seqs = beam_seqs.t().repeat(self.opt.eval.bs, 1).t().contiguous().view(
                self.opt.eval.bs**2, -1)[top_beam_idxs]
            beam_seqs = torch.cat((beam_seqs, beam_toks.unsqueeze(1)), dim=1)

            # I have no idea what's going on but Ari's on point with it
            XMB = XMB.transpose(0, 1).transpose(1, 2).repeat(
                self.opt.eval.bs, 1, 1).transpose(2, 1).transpose(
                1, 0).contiguous().view(
                self.opt.eval.bs**2, XMB.size(1), XMB.size(2))[top_beam_idxs]

            XMB, MMB = self.append_batch(XMB, beam_toks, MMB)

            if (beam_toks == self.end_token).sum().item() == self.opt.eval.bs:
                break

        beams = []

        for beam in beam_seqs:
            beams.append(" ".join("".join(
                [data_loader.vocab_decoder[tok.item()].replace(
                    '</w>', ' ').replace('\n', '')
                 for tok in beam if tok != self.end_token]).split()))

        sampling_result = {
            "sequence": beams[0],
            "beams": beams,
            "beam_losses": beam_lls.tolist(),
            "loss": beam_lls[0].item(),
            "beam_lengths": counts.tolist(),
            "length": counts[0].item()
        }

        return sampling_result
