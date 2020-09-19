from random import randint, shuffle
from random import random as rand
import pickle
import json
from collections import namedtuple
import torch
import torch.nn as nn
import unicodedata
from multiprocessing import Lock


def get_random_word(vocab_words):
    i = randint(0, len(vocab_words) - 1)
    return vocab_words[i]


def batch_list_to_batch_tensors(batch):
    batch_tensors = []
    for x in zip(*batch):
        if isinstance(x[0], torch.Tensor):
            batch_tensors.append(torch.stack(x))
        else:
            batch_tensors.append(torch.tensor(x, dtype=torch.long))
    return batch_tensors


def batch_list_to_batch_tensors_rank_loss(batch):
    assert len(batch) == 1
    batch_tensors = []
    for x in zip(*batch[0]):
        if isinstance(x[0], torch.Tensor):
            batch_tensors.append(torch.stack(x))
        else:
            batch_tensors.append(torch.tensor(x, dtype=torch.long))
    return batch_tensors


# (input_ids, segment_ids, input_mask, masked_ids, masked_pos, masked_weights, nsp_label, img, vis_pe)

def batch_list_to_batch_tensors_truncate(batch):
    batch_tensors = []
    max_len_in_batch = max([len(b[0]) for b in batch])
    print("\nMax len in batch: %d! \n" % max_len_in_batch)

    for idx, x in enumerate(zip(*batch)):
        if idx == 0 or idx == 1:
            for x_i in x:
                n_pad = max_len_in_batch - len(x_i)
                x_i.extend([0] * n_pad)
            batch_tensors.append(torch.tensor(x, dtype=torch.long))
        elif idx == 2:
            assert isinstance(x[0], torch.Tensor)
            xs = [x_i[:max_len_in_batch, :max_len_in_batch] for x_i in x]
            batch_tensors.append(torch.stack(xs))
        else:
            if isinstance(x[0], torch.Tensor):
                batch_tensors.append(torch.stack(x))
            else:
                batch_tensors.append(torch.tensor(x, dtype=torch.long))
    return batch_tensors


class Pipeline():
    """ Pre-process Pipeline Class : callable """

    def __init__(self):
        super().__init__()
        self.mask_same_word = None
        self.skipgram_prb = None
        self.skipgram_size = None

    def __call__(self, instance):
        raise NotImplementedError
