import math
import random

import torch


class Dataset:

    def __init__(self, samples, batch_size):
        self.samples = samples
        self.batch_size = batch_size
        self.num_batches = math.ceil(len(self.samples) / batch_size)
        self.speaker_to_idx = {'M': 0, 'F': 1}

    def __len__(self):
        return self.num_batches

    def __getitem__(self, index):
        batch = self.raw_batch(index)
        return self.padding(batch)

    def raw_batch(self, index):
        assert index < self.num_batches, "batch_idx %d > %d" % (index, self.num_batches)
        batch = self.samples[index * self.batch_size: (index + 1) * self.batch_size]

        return batch

    def padding(self, samples):
        batch_size = len(samples)
        text_len_tensor = torch.tensor([len(s.text) for s in samples]).long()
        mx = torch.max(text_len_tensor).item()
        text_tensor = torch.zeros((batch_size, mx, 100))
        speaker_tensor = torch.zeros((batch_size, mx)).long()
        labels = []
        for i, s in enumerate(samples):
            cur_len = len(s.text)
            tmp = [torch.from_numpy(t).float() for t in s.text]
            tmp = torch.stack(tmp)
            text_tensor[i, :cur_len, :] = tmp
            speaker_tensor[i, :cur_len] = torch.tensor([self.speaker_to_idx[c] for c in s.speaker])
            labels.extend(s.label)

        label_tensor = torch.tensor(labels).long()
        data = {
            "text_len_tensor": text_len_tensor,
            "text_tensor": text_tensor,
            "speaker_tensor": speaker_tensor,
            "label_tensor": label_tensor
        }

        return data

    def shuffle(self):
        random.shuffle(self.samples)




