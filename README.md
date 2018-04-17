# End-To-End Memory Networks in PyTorch

This repo is the PyTorch implementation of MemN2N model proposed in [End-To-End Memory Networks](https://arxiv.org/abs/1503.08895v5) and focused on the section 4 -  Synthetic Question and Answering Experiments - of the original paper.

![Imgur](https://i.imgur.com/kJpX8Dk.png)



## Requirements

+ [PyTorch](http://pytorch.org/) >=0.3.1
+ [torchtext](https://github.com/shihan9/text) (you need to install my fork version of torchtext since the PR hasn't been accepted yet.)
+ [click](http://click.pocoo.org/5/) >=6.7

## Dataset

The dataset is bAbI 20 QA tasks (v1.2) from [facebook research](https://research.fb.com/), which you can find it [here](https://research.fb.com/downloads/babi/).

## Benchmarks

| Task                     | BoW AD 3HOPS | PE AD 3HOPS | PE AD 3HOPS JOINT |
| ------------------------ | ------------ | ----------- | ----------------- |
| 1: 1 supporting fact     | 0.1          | 0.0         | 1.3               |
| 2: 2 supporting facts    | 48.4         | 16.1        | 51.1              |
| 3: 3 supporting facts    | 75.9         | 74          | 57.3              |
| 4: 2 argument relations  | 31.7         | 0.6         | 5.7               |
| 5: 3 argument relations  | 19.5         | 13.8        | 38.7              |
| 6: yes/no questions      | 6.6          | 9.6         | 8.2               |
| 7: counting              | 21.1         | 18.7        | 45.9              |
| 8: lists/sets            | 15.0         | 12.7        | 40.9              |
| 9: simple negation       | 11.5         | 7.9         | 8.0               |
| 10: indeﬁnite knowledge  | 14.6         | 14.0        | 28.3              |
| 11: basic coreference    | 16.6         | 4.8         | 14.6              |
| 12: conjunction          | 0.0          | 0.0         | 4.5               |
| 13: compound coreference | 8.9          | 7.0         | 21.7              |
| 14: time reasoning       | 28.1         | 8.4         | 42.9              |
| 15: basic deduction      | 49.5         | 0.0         | 23.0              |
| 16: basic induction      | 55.4         | 55.1        | 56.8              |
| 17: positional reasoning | 49.2         | 46.9        | 43                |
| 18: size reasoning       | 44.7         | 8.5         | 12.3              |
| 19: path ﬁnding          | 90.0         | 82.6        | 90.9              |
| 20: agent’s motivation   | 0.1          | 0.3         | 0.2               |

## Usage

To train by default setting:

```shell
python cli.py --train
```

To see all training options:

```shell
python cli.py --help

Usage: cli.py [OPTIONS]

Options:
  --train                    Train phase.
  -s, --save_dir TEXT        Directory of saved object files.
                             [default: .save]
  -f, --file TEXT            Path of saved object file to load.
  -o, --num_epochs INTEGER   Number of epochs to train.
                             [default: 100]
  -b, --batch_size INTEGER   Batch size.  [default: 32]
  --lr FLOAT                 Learning rate.  [default: 0.02]
  -e, --embed_size INTEGER   Embedding size.  [default: 20]
  -t, --task INTEGER         Number of task to learn.
                             [default: 1]
  -m, --memory_size INTEGER  Capacity of memory.  [default: 50]
  -h, --num_hops INTEGER     Embedding size.  [default: 3]
  -c, --max_clip FLOAT       Max gradient norm to clip
                             [default: 40.0]
  -j, --joint                Joint learning.
  -k, --tenk                 Use 10K dataset.
  -w, --use_bow              Use BoW, or PE sentence
                             representation.
  -l, --use_lw               Use layer-wise, or adjacent weight
                             tying.
  --help                     Show this message and exit.
```



## TODOs

+ Linear Start
+ Random Noise
