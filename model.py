import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I


class MemN2N(nn.Module):

    def __init__(self, params, vocab):
        super(MemN2N, self).__init__()
        self.input_size = len(vocab)
        self.embed_size = params.embed_size
        self.memory_size = params.memory_size
        self.num_hops = params.num_hops
        self.use_bow = params.use_bow
        self.use_lw = params.use_lw
        self.use_ls = params.use_ls
        self.vocab = vocab

        # create parameters according to different type of weight tying
        pad = self.vocab.stoi['<pad>']
        self.A = nn.ModuleList([nn.Embedding(self.input_size, self.embed_size, padding_idx=pad)])
        self.A[-1].weight.data.normal_(0, 0.1)
        self.C = nn.ModuleList([nn.Embedding(self.input_size, self.embed_size, padding_idx=pad)])
        self.C[-1].weight.data.normal_(0, 0.1)
        if self.use_lw:
            for _ in range(1, self.num_hops):
                self.A.append(self.A[-1])
                self.C.append(self.C[-1])
            self.B = nn.Embedding(self.input_size, self.embed_size, padding_idx=pad)
            self.B.weight.data.normal_(0, 0.1)
            self.out = nn.Parameter(
                I.normal_(torch.empty(self.input_size, self.embed_size), 0, 0.1))
            self.H = nn.Linear(self.embed_size, self.embed_size)
            self.H.weight.data.normal_(0, 0.1)
        else:
            for _ in range(1, self.num_hops):
                self.A.append(self.C[-1])
                self.C.append(nn.Embedding(self.input_size, self.embed_size, padding_idx=pad))
                self.C[-1].weight.data.normal_(0, 0.1)
            self.B = self.A[0]
            self.out = self.C[-1].weight

        # temporal matrix
        self.TA = nn.Parameter(I.normal_(torch.empty(self.memory_size, self.embed_size), 0, 0.1))
        self.TC = nn.Parameter(I.normal_(torch.empty(self.memory_size, self.embed_size), 0, 0.1))

    def forward(self, story, query):
        sen_size = query.shape[-1]
        weights = self.compute_weights(sen_size)
        state = (self.B(query) * weights).sum(1)

        sen_size = story.shape[-1]
        weights = self.compute_weights(sen_size)
        for i in range(self.num_hops):
            memory = (self.A[i](story.view(-1, sen_size)) * weights).sum(1).view(
                *story.shape[:-1], -1)
            memory += self.TA
            output = (self.C[i](story.view(-1, sen_size)) * weights).sum(1).view(
                *story.shape[:-1], -1)
            output += self.TC

            probs = (memory @ state.unsqueeze(-1)).squeeze()
            if not self.use_ls:
                probs = F.softmax(probs, dim=-1)
            response = (probs.unsqueeze(1) @ output).squeeze()
            if self.use_lw:
                state = self.H(response) + state
            else:
                state = response + state

        return F.log_softmax(F.linear(state, self.out), dim=-1)

    def compute_weights(self, J):
        d = self.embed_size
        if self.use_bow:
            weights = torch.ones(J, d)
        else:
            func = lambda j, k: 1 - (j + 1) / J - (k + 1) / d * (1 - 2 * (j + 1) / J)    # 0-based indexing
            weights = torch.from_numpy(np.fromfunction(func, (J, d), dtype=np.float32))
        return weights.cuda() if torch.cuda.is_available() else weights
