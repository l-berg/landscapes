from src.utils.utils import set_seed, save_data, load_data, get_model, get_dataloaders

import torch
import numpy as np

import os
import argparse
from pathlib import Path
from tqdm import tqdm


parser = argparse.ArgumentParser(description='''
Compute grid of losses over two vectors that perturb the model's parameters.
''')

parser.add_argument('grid_name')
parser.add_argument('model_type', choices=['resnet', 'vgg'])
parser.add_argument('dataset', choices=['cifar-10', 'tiny-imagenet', 'fashion-mnist', 'mnist'])
parser.add_argument('--layer', default='all', help="specify which layer gets perturbed")
parser.add_argument("--ngrids", type=int, default=3,
                    help="number of different grids use (= #pairs of perturbation vectors)")
parser.add_argument("--nbatches", type=int, default=3, help="number of data-batches to compute losses for")
parser.add_argument("--batch_size", type=int, default=256, help="batch size")
parser.add_argument("--step_scale", type=float, default=1.0, help="scales distance between grid points")
parser.add_argument("--grid_width", type=int, default=10, help="size of grid with width=height")
# parser.add_argument('--load', dest='load', default=False, action='store_true', help="load grids instead of computing")

parser.add_argument('--list-layers', dest='list_layers', default=False, action='store_true',
                    help="print a list of the model's layers and exit")

parser.add_argument("--nepochs", type=int, default=10, help="number of epochs")
parser.add_argument("--nworkers", type=int, default=4, help="number of workers for dataloaders")
parser.add_argument("--seed", type=int, default=0, help="random seed")
# parser.add_argument('--checkpoint_every', type=int, default=1000)
# parser.add_argument('--checkpoint')
args = parser.parse_args()

# device = 'cpu'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using {}'.format(device))


class GridWalk:
    def __init__(self, params, vector_pair, step_size, grid_width):
        # backup original parameter values
        self.backup_params = []
        for p in params:
            self.backup_params.append(p.data.detach())

        # scale perturbation vectors once, now
        self.vecs1 = [v * step_size for v in vector_pair[0]]
        self.vecs2 = [v * step_size for v in vector_pair[1]]

        self.params = params
        self.grid_width = grid_width
        self.grid = []
        self.current_row = []

        # start in corner of grid
        self._set_params(0, 0)

    def reset(self):
        self.grid = []
        self.current_row = []
        self._set_params(0, 0)

    @torch.no_grad()
    def _set_params(self, cix, rix):
        for p, root, v1, v2 in zip(self.params, self.backup_params, self.vecs1, self.vecs2):
            p.data = root + (cix - (self.grid_width-1)/2) * v1 + (rix - (self.grid_width-1)/2) * v2

    def step(self, grid_value):
        """Adjusts model parameters and returns True if walk is done"""

        # store given value
        self.current_row.append(grid_value)
        if len(self.current_row) >= self.grid_width:
            self.grid.append(self.current_row)
            self.current_row = []

        # update model parameters
        cix = len(self.grid)
        rix = len(self.current_row)
        self._set_params(cix, rix)

        done = len(self.grid) >= self.grid_width
        if done:
            # restore original state of model
            for p, root in zip(self.params, self.backup_params):
                p.data = root

        return done


@torch.no_grad()
def sample_grid(model, loader, criterion, grid_walk, desc=''):
    model.train()  # we don't really want to train, but analyze the state

    with tqdm(total=args.grid_width**2) as pbar:
        pbar.set_description(desc)

        done = False
        while not done:
            losses = []

            # run all batches for the current model version
            for i, (X, y) in zip(range(args.nbatches), loader):
                X, y = X.to(device), y.to(device)

                output = model(X)
                _, pred = torch.max(output, 1)
                loss = criterion(output, y)
                losses.append(loss)

            done = grid_walk.step([l.item() for l in losses])
            pbar.update(1)


def average_grad_std(model, loader, criterion):
    """Inspect gradients of parameters"""
    grad_stds = []

    model.train()
    for i, (X, y) in zip(range(args.nbatches), loader):
        X, y = X.to(device), y.to(device)

        output = model(X)
        _, pred = torch.max(output, 1)
        loss = criterion(output, y)

        loss.backward()

        all_grads = []
        for p in model.parameters():
            if p.grad is not None:
                all_grads.append(p.grad.view(-1))
        all_grads = torch.cat(all_grads)
        grad_stds.append(all_grads.std().item())

    return sum(grad_stds) / len(grad_stds)


def main():
    args_path = os.path.join('results', args.dataset, args.model_type)
    model_path = os.path.join(args_path, 'training_run/1/model.pt')
    vectors_path = os.path.join(args_path, 'training_run/vectors', args.grid_name)
    grids_path = os.path.join(args_path, 'training_run/grids', args.grid_name)

    # create missing directories if necessary
    Path(os.path.dirname(vectors_path)).mkdir(exist_ok=True)
    Path(os.path.dirname(grids_path)).mkdir(exist_ok=True)

    # load dataset and model
    train_loader, val_loader = get_dataloaders(args.dataset, batch_size=args.batch_size,
                                               nworkers=args.nworkers, shuffle=False)
    _, in_channels, in_width, _ = next(iter(train_loader))[0].shape
    model = get_model(args.model_type, in_channels, in_width).to(device)
    model.load_state_dict(torch.load(model_path))
    criterion = torch.nn.CrossEntropyLoss()

    if args.list_layers:
        # print info about model and layers
        print(f'\nThe {args.model_type} model for {args.dataset}:\n{str(model)}')
        print('\nThe model has the following parameters:')
        for name, parameter in model.named_parameters():
            print(f'\t{name}')
        return

    if args.layer == 'all':
        # create perturbation vectors, scale using std of gradient
        if os.path.exists(vectors_path):
            vectors = load_data(vectors_path)
            print(f'Loaded {len(vectors)} perturbation vector pairs.')
        else:
            vectors = []
            for _ in range(args.ngrids):
                vec1, vec2 = {}, {}
                for name, parameter in model.named_parameters():
                    vec1[name] = torch.randn_like(parameter).detach()
                    vec2[name] = torch.randn_like(parameter).detach()
                vectors.append((vec1, vec2))
            save_data(vectors_path, vectors)
            print(f'Created {len(vectors)} perturbation vector pairs.')

        step_size = 0.01  # 0.001 * average_grad_std(model, train_loader, criterion)

        # all_grids is a matrix of loss landscapes:
        #       batch1, batch2, batch3, ...
        # pair1   x       x       x     ...
        # pair2   x       x       x     ...
        # pair3   x       x       x     ...
        # ...
        all_grids = []
        for pair_no, pair in enumerate(vectors):
            pair_parameters = [[v for k, v in vec.items()] for vec in pair]
            grid_walk = GridWalk(model.parameters(), pair_parameters, step_size, args.grid_width)
            sample_grid(model, train_loader, criterion, grid_walk, desc=f"Permutation {pair_no+1}/{len(vectors)}")
            grids = np.array(grid_walk.grid)

            # reorder axes from (v1, v2, batch) to (batch, v1, v2)
            grids = np.moveaxis(grids, -1, 0)
            grids = list(grids)  # batch-wise should be normal python list
            all_grids.append(grids)
        save_data(grids_path, all_grids)
        print(all_grids)

    # import ipdb; ipdb.set_trace()

if __name__ == '__main__':
    set_seed(args.seed)  # to get reproducible results
    main()
