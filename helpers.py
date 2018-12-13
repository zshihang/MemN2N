from collections import namedtuple

import torch
from torchtext.datasets import BABI20


def dataloader(batch_size, memory_size, task, joint, tenK):
    train_iter, valid_iter, test_iter = BABI20.iters(
        batch_size=batch_size, memory_size=memory_size, task=task, joint=joint, tenK=tenK, device=torch.device("cpu"))
    return train_iter, valid_iter, test_iter, train_iter.dataset.fields['query'].vocab


def get_params(config):
    Params = namedtuple('Params', [
        'embed_size',
        'memory_size',
        'num_hops',
        'use_bow',
        'use_lw',
        'use_ls',
    ])
    params = Params(
        config.embed_size, config.memory_size, config.num_hops, config.use_bow,
        config.use_lw, config.use_ls)
    return params


def get_fname(config):
    fname = "_".join(
        str(x) for x in [
            config.num_epochs,
            config.lr,
            config.batch_size,
            config.embed_size,
            config.memory_size,
            config.num_hops,
            config.joint,
            config.tenk,
            config.use_bow,
            config.use_lw,
            config.use_ls,
        ]) + '.task' + str(config.task)
    return fname
