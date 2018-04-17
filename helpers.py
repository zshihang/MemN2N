from collections import namedtuple

from torchtext.datasets import BABI20


def dataloader(batch_size, memory_size, task, joint, tenK):
    train_iter, _, test_iter = BABI20.iters(batch_size=batch_size, memory_size=memory_size,
                                            task=task, joint=joint, tenK=tenK)
    return train_iter, test_iter, train_iter.dataset.fields['query'].vocab


def get_params(config, vocab):
    Params = namedtuple(
        'Params',
        ['input_size', 'embed_size', 'memory_size', 'num_hops', 'use_bow', 'use_lw', 'vocab'])
    params = Params(
        len(vocab), config.embed_size, config.memory_size, config.num_hops, config.use_bow,
        config.use_lw, vocab)
    return params


def get_fname(config):
    fname = "_".join(
        str(x) for x in [
            config.num_epochs, config.lr, config.batch_size, config.embed_size, config.memory_size,
            config.num_hops, config.joint, config.tenk, config.use_bow, config.use_lw
        ]) + '.task' + str(config.task)
    return fname
