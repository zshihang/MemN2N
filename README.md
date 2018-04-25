# End-To-End Memory Networks in PyTorch

This repo is the PyTorch implementation of MemN2N model proposed in [End-To-End Memory Networks](https://arxiv.org/abs/1503.08895v5) and focused on the section 4 -  Synthetic Question and Answering Experiments - of the original paper.

![Imgur](https://i.imgur.com/kJpX8Dk.png)



## Requirements

+ [PyTorch](http://pytorch.org/) ==0.4.0
+ [torchtext](https://github.com/shihan9/text) (please install my fork version of torchtext since the PR hasn't been accepted yet.)
+ [click](http://click.pocoo.org/5/) ==6.7

## Dataset

The dataset is bAbI 20 QA tasks (v1.2) from [facebook research](https://research.fb.com/), which you can find it [here](https://research.fb.com/downloads/babi/).

## Benchmarks

| Task                     | BoW 3HOPS | PE 3HOPS | PE 3HOPS JOINT | PE LS 3HOPS JOINT |
| ------------------------ | --------- | -------- | -------------- | ----------------- |
| 1: 1 supporting fact     | 0.1       | 0.0      | 1.3            | 0.2               |
| 2: 2 supporting facts    | 48.4      | 16.1     | 51.1           | 19.7              |
| 3: 3 supporting facts    | 75.9      | 74       | 57.3           | 32.7              |
| 4: 2 argument relations  | 31.7      | 0.6      | 5.7            | 2.0               |
| 5: 3 argument relations  | 19.5      | 13.8     | 38.7           | 11.6              |
| 6: yes/no questions      | 6.6       | 9.6      | 8.2            | 1.7               |
| 7: counting              | 21.1      | 18.7     | 45.9           | 20.3              |
| 8: lists/sets            | 15.0      | 12.7     | 40.9           | 16.1              |
| 9: simple negation       | 11.5      | 7.9      | 8.0            | 3.0               |
| 10: indeﬁnite knowledge  | 14.6      | 14.0     | 28.3           | 12.7              |
| 11: basic coreference    | 16.6      | 4.8      | 14.6           | 14.2              |
| 12: conjunction          | 0.0       | 0.0      | 4.5            | 1.8               |
| 13: compound coreference | 8.9       | 7.0      | 21.7           | 11.9              |
| 14: time reasoning       | 28.1      | 8.4      | 42.9           | 7.3               |
| 15: basic deduction      | 49.5      | 0.0      | 23.0           | 1.3               |
| 16: basic induction      | 55.4      | 55.1     | 56.8           | 56.2              |
| 17: positional reasoning | 49.2      | 46.9     | 43             | 41.2              |
| 18: size reasoning       | 44.7      | 8.5      | 12.3           | 8.1               |
| 19: path ﬁnding          | 90.0      | 82.6     | 90.9           | 89.0              |
| 20: agent’s motivation   | 0.1       | 0.3      | 0.2            | 0.1               |

All the results are for 1k  training set and picked from multiple runs with the same parameter settings. Key: BoW = bag-of-words representation; PE = position encoding representation; LS = linear start training; joint = joint training on all tasks (as opposed to per-task training); adjacent weight tying is used.

Notes:

+ For per-task training, shuffling the data for every epoch helps (better results compared to the original paper); For joint training, shuffling all the data worsen the results. Based these two observations, it is reasonable to foresee that shuffling within tasks rather than among tasks in the joint training setting will help to reduce error.
+ Joint training on all tasks indeed helps.
+ Tried training task 16 using linear start and PE, but did not see the sharp drop to lower than 5 from the original paper.
+ The position encoding (PE) representation beats BoW on task 2, 4, 5, 15, 18. (original paper doesn't show this on task 2)
+ Several tasks are very sensitive to initializations, e.g. task 2.
+ Linear start plays a significant role in joint learning.

## Usage

To train by default setting:

```shell
python cli.py --train
```

To see all training options:

```shell
Usage: cli.py [OPTIONS]

Options:
  --train                Train phase.
  --save_dir TEXT        Directory of saved object files.  [default: .save]
  --file TEXT            Path of saved object file to load.
  --num_epochs INTEGER   Number of epochs to train.  [default: 100]
  --batch_size INTEGER   Batch size.  [default: 32]
  --lr FLOAT             Learning rate.  [default: 0.02]
  --embed_size INTEGER   Embedding size.  [default: 20]
  --task INTEGER         Number of task to learn.  [default: 1]
  --memory_size INTEGER  Capacity of memory.  [default: 50]
  --num_hops INTEGER     Embedding size.  [default: 3]
  --max_clip FLOAT       Max gradient norm to clip  [default: 40.0]
  --joint                Joint learning.
  --tenk                 Use 10K dataset.
  --use_bow              Use BoW, or PE sentence representation.
  --use_lw               Use layer-wise, or adjacent weight tying.
  --use_ls               Use linear start.
  --help                 Show this message and exit.
```



## TODOs

+ ~~Linear Start~~
+ Random Noise

