from src.utils.utils import set_seed, save_data, load_data, get_model, get_dataloaders

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch

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

#def plot_gradient(grid, ax, gradient, pair, step_size):
#    y_root, x_root = [(s-1)/2 for s in grid.shape]
#
#    param_names = sorted(pair[0])
#    pair_in_one = [step_size * torch.cat([v[name].view(-1) for name in param_names]) for v in pair]
#    gradient_in_one = torch.cat([gradient[name].view(-1) for name in param_names])
#
#    dy, dx = [(torch.dot(v, gradient_in_one) / torch.linalg.norm(v)).item() for v, s in zip(pair_in_one, grid.shape)]
#    dx, dy = np.array([dx, dy]) / np.linalg.norm(np.array([dx, dy])) * len(grid) * 0.12
#
#    ax.arrow(x_root, y_root, dx, dy, head_width=len(grid)*0.03, head_length=len(grid)*0.03, fc='k', ec='k', length_includes_head=True)

def project_gradients(vectors, gradients, step_size, desired_length=2.0):
    learning_rate = 0.001  # when training with SGD
    param_names = sorted(vectors[0][0])
    vectors_in_one = [[torch.cat([v[name].view(-1) for name in param_names]) for v in pair] for pair in vectors]
    gradients_in_one = [learning_rate * torch.cat([g[name].view(-1) for name in param_names]) for g in gradients]
    arrow_grid = []
    for pair in vectors_in_one:
        arrow_row = []
        for grad in gradients_in_one:
            dy, dx = [(torch.dot(v, grad) / torch.linalg.norm(v)).item() for v in pair]
            arrow_row.append(np.array([dx, dy]))
        arrow_grid.append(arrow_row)

    lengths = [np.linalg.norm(a) for line in arrow_grid for a in line]
    max_length = max(lengths)
    scale_factor = desired_length / max_length

    arrow_grid = [[scale_factor * a for a in row] for row in arrow_grid]

    return arrow_grid, scale_factor * step_size

@torch.no_grad()
def main():
    args_path = os.path.join('results', args.dataset, args.model_type)
    grids_path = os.path.join(args_path, 'training_run/grid_data', args.grid_name)
    if not os.path.exists(grids_path):
        print(f'Error: no grids found at {grids_path}. Compute them first with grids.py')
        return
    img_path = os.path.join(args_path, 'training_run/images', f'{args.grid_name}.pdf')
    Path(os.path.dirname(img_path)).mkdir(exist_ok=True)

    grids, vectors, step_size, gradients = load_data(grids_path)

    # project gradient onto plane spanned by vectors
    arrow_grid, arrow_scale_factor = project_gradients(vectors, gradients, step_size, desired_length=len(grids[0][0])*0.25)

    # remove mean over every batch
    for batch_no in range(len(grids[0])):
        batch_mean = np.array([line[batch_no].mean() for line in grids]).mean()
        for line in grids:
            line[batch_no] -= batch_mean

    # some plot parameters
    cmap = 'viridis'
    vmin = np.array(grids).min()
    vmax = np.array(grids).max()
    y_root, x_root = [(s-1)/2 for s in grids[0][0].shape]

    # plot all grids
    fig, ax = plt.subplots(nrows=len(grids), ncols=len(grids[0]), figsize=(8,8), squeeze=False)
    fig.suptitle(f'Loss landscapes. SGD-update projections scaled by {arrow_scale_factor:.2e}')
    for pair_no, batch_of_grids in enumerate(grids):
        for batch_no, grid in enumerate(batch_of_grids):
            a = ax[pair_no, batch_no]
            plot_grid(grid, a, cmap=cmap, vmin=vmin, vmax=vmax)

            # draw gradient arrow
            #plot_gradient(grid, a, gradients[batch_no], vectors[pair_no], step_size)
            dx, dy = arrow_grid[pair_no][batch_no]
            a.arrow(x_root, y_root, dx, dy, head_width=len(grid) * 0.03, head_length=len(grid) * 0.03,
                    fc='k', ec='k', length_includes_head=True)

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

    fig.savefig(img_path, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    main()
