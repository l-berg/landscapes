from src.utils.utils import set_seed, save_data, load_data, get_model, get_dataloaders

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import os
import argparse
from pathlib import Path


parser = argparse.ArgumentParser(description='''
Compute grid of losses over two vectors that perturb the model's parameters.
''')

parser.add_argument('grid_name')
parser.add_argument('model_type', choices=['resnet', 'vgg'])
parser.add_argument('dataset', choices=['cifar-10', 'tiny-imagenet', 'fashion-mnist', 'mnist'])
parser.add_argument('args', nargs=argparse.REMAINDER)

args = parser.parse_args()


def heatmap2d(arr: np.ndarray):
    plt.imshow(arr, cmap='viridis')
    plt.colorbar()
    plt.show()

def plot_grid(grid, ax, **kwargs):
    ax.tick_params(axis='both', which='both',
                   bottom=False, top=False, left=False, right=False,
                   labelbottom=False, labelleft=False)
    ax.imshow(grid, interpolation='nearest', **kwargs)

def main():
    args_path = os.path.join('results', args.dataset, args.model_type)
    grids_path = os.path.join(args_path, 'training_run/grids', args.grid_name)
    if not os.path.exists(grids_path):
        print(f'Error: no grids found at {grids_path}. Compute them first with grids.py')
        return

    grids = load_data(grids_path)

    # some plot parameters
    cmap = 'viridis'
    vmin = np.array(grids).min()
    vmax = np.array(grids).max()

    # plot all grids
    fig, ax = plt.subplots(nrows=len(grids), ncols=len(grids[0]), figsize=(8,8))
    for pair_no, batch_of_grids in enumerate(grids):
        for batch_no, grid in enumerate(batch_of_grids):
            a = ax[pair_no, batch_no]
            plot_grid(grid, a, cmap=cmap, vmin=vmin, vmax=vmax)
            if pair_no == 0:
                a.xaxis.set_label_position('top')
                a.set_xlabel(f'batch {batch_no+1}')
            if batch_no == 0:
                a.set_ylabel(f'perturbation {pair_no+1}')

    # add colorbar
    plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
    cax = plt.axes([0.85, 0.1, 0.075, 0.8])
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm)

    plt.show()


if __name__ == '__main__':
    main()
