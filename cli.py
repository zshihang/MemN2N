from collections import namedtuple

import click

from main import run


@click.command()
@click.option('--train', is_flag=True, help="Train phase.")
@click.option('-s', '--save_dir', default='.save', help="Directory of saved object files.",
              show_default=True)
@click.option('-f', '--file', default='', help="Path of saved object file to load.")
@click.option('-o', '--num_epochs', type=int, default=100, help="Number of epochs to train.",
              show_default=True)
@click.option('-b', '--batch_size', type=int, default=32, help="Batch size.", show_default=True)
@click.option('--lr', type=float, default=0.02, help="Learning rate.", show_default=True)
@click.option('-e', '--embed_size', type=int, default=20, help="Embedding size.",
              show_default=True)
@click.option('-t', '--task', type=int, default=1, help="Number of task to learn.",
              show_default=True)
@click.option('-m', '--memory_size', type=int, default=50, help="Capacity of memory.",
              show_default=True)
@click.option('-h', '--num_hops', type=int, default=3, help="Embedding size.", show_default=True)
@click.option('-c', '--max_clip', type=float, default=40.0, help="Max gradient norm to clip",
              show_default=True)
@click.option('-j', '--joint', is_flag=True, help="Joint learning.")
@click.option('-k', '--tenk', is_flag=True, help="Use 10K dataset.")
@click.option('-w', '--use_bow', is_flag=True, help="Use BoW, or PE sentence representation.")
@click.option('-l', '--use_lw', is_flag=True, help="Use layer-wise, or adjacent weight tying.")
def cli(**kwargs):
    config = namedtuple("Config", kwargs.keys())(**kwargs)
    run(config)


if __name__ == "__main__":
    cli()
