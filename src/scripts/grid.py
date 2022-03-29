from src.utils.utils import set_seed, save_data, load_data, get_model, get_dataloaders, get_args_path

import torch
import numpy as np

import re
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
parser.add_argument('--activation', default='relu', choices=['relu', 'sigmoid', 'tanh'],
                    help="make sure to keep this consistent between calls to train, grid and visualize")
parser.add_argument('--layer', default='all', help="specify which layer gets perturbed")
parser.add_argument("--ngrids", type=int, default=3,
                    help="number of different grids use (= #pairs of perturbation vectors)")
parser.add_argument("--nbatches", type=int, default=3, help="number of data-batches to compute losses for")
parser.add_argument("--batch_size", type=int, default=256, help="batch size")
parser.add_argument("--max_step", type=float, default=1.0, help="distance to edge of grid")
#parser.add_argument("--step_scale", type=float, default=1.0, help="scales distance between grid points")
parser.add_argument("--grid_width", type=int, default=11, help="size of grid with width=height")
parser.add_argument('--use_perturbations_from', help="use the same perturbation vectors as another grid")
parser.add_argument('--single_kernel', dest='single_kernel', default=False, action='store_true',
                    help="only perturb the first kernel of the specified conv layer")
# parser.add_argument('--load', dest='load', default=False, action='store_true', help="load grids instead of computing")
parser.add_argument("--episode", type=int, default=1, help="uses model checkpoint after this episode")
parser.add_argument('--small_gpu', dest='small_gpu', default=False, action='store_true',
                    help="do parameter modifications on CPU to save GPU-RAM")

parser.add_argument('--list_layers', dest='list_layers', default=False, action='store_true',
                    help="print a list of the model's layers and exit")

parser.add_argument("--nworkers", type=int, default=4, help="number of workers for dataloaders")
parser.add_argument("--seed", type=int, default=0, help="random seed")
args = parser.parse_args()

# device = 'cpu'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using {}'.format(device))


class GridWalk:
    def __init__(self, params, vector_pair, step_size, grid_width):
        params = list(params)

        # backup original parameter values
        self.backup_params = []
        for p in params:
            if args.small_gpu:
                self.backup_params.append(p.data.detach().cpu())
            else:
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
        #import ipdb; ipdb.set_trace()
        for p, root, v1, v2 in zip(self.params, self.backup_params, self.vecs1, self.vecs2):
            if args.small_gpu:
                p.data = (root + (cix - (self.grid_width-1)/2) * v1 + (rix - (self.grid_width-1)/2) * v2).to(device)
            else:
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


def gather_gradients(model, loader, criterion):
    """Returns a list of gradient vectors for the different batches."""
    gradients = []

    model.train()
    for i, (X, y) in zip(range(args.nbatches), loader):
        print(i)
        X, y = X.to(device), y.to(device)

        output = model(X)
        _, pred = torch.max(output, 1)
        loss = criterion(output, y)

        loss.backward()

        with torch.no_grad():
            gradient_vec = {}
            for name, parameter in model.named_parameters():
                gradient_vec[name] = parameter.grad.clone().detach().cpu()
            gradients.append(gradient_vec)

    return gradients

def average_grad_std(gradients):
    """Return average grad std over all batches."""
    grad_stds = []

    for vec in gradients:
        all_grads = []
        for name, grad in vec.items():
            if grad is not None:
                all_grads.append(grad.view(-1))
        all_grads = torch.cat(all_grads)
        grad_stds.append(all_grads.std().item())

    return sum(grad_stds) / len(grad_stds)


def random_direction(weights):
    # filter-normalization from https://arxiv.org/abs/1712.09913
    filter_stds = weights.std(dim=list(range(1, weights.ndim)), keepdim=True)
    return filter_stds * torch.randn_like(weights)  # broadcasting to whole filter


def main():
    args_path = get_args_path(args.dataset, args.model_type, args.activation)
    model_path = os.path.join(args_path, f'training_run/{args.episode}/model.pt')
    grids_path = os.path.join(args_path, 'training_run/grid_data', args.grid_name)

    # use another grid's perturbations
    if not args.use_perturbations_from:
        vectors_path = os.path.join(args_path, 'training_run/vectors', args.grid_name)
    else:
        vectors_path = os.path.join(args_path, 'training_run/vectors', args.use_perturbations_from)
        if not os.path.exists(vectors_path):
            raise FileNotFoundError(f'Grid with name {args.use_perturbations_from} not found')

    # create missing directories if necessary
    Path(os.path.dirname(vectors_path)).mkdir(exist_ok=True)
    Path(os.path.dirname(grids_path)).mkdir(exist_ok=True)

    # load dataset and model
    train_loader, val_loader = get_dataloaders(args.dataset, batch_size=args.batch_size,
                                               nworkers=args.nworkers, shuffle=False)
    _, in_channels, in_width, _ = next(iter(train_loader))[0].shape
    model = get_model(args.model_type, in_channels, in_width, activation=args.activation).to(device)
    model.load_state_dict(torch.load(model_path))
    criterion = torch.nn.CrossEntropyLoss()

    if args.list_layers:
        # print info about model and layers
        print(f'\nThe {args.model_type} model for {args.dataset}:\n{str(model)}')
        print('\nThe model has the following parameters:')
        for name, parameter in model.named_parameters():
            print(f'\t{name}')
        return

    # create perturbation vectors
    if os.path.exists(vectors_path):
        vectors = load_data(vectors_path)
        # make sure everything is on the correct device
        for pair in vectors:
            for v in pair:
                for name in v:
                    v[name] = v[name].to(device)
        print(f'Loaded {len(vectors)} perturbation vector pairs from {vectors_path}.')
    else:
        bn_regex = re.compile(r"bn[0-9]*\.(weight|bias)$")
        vectors = []
        for _ in range(args.ngrids):

            vec1, vec2 = {}, {}
            for name, parameter in model.named_parameters():
                # skip batch-norm params
                if bn_regex.match(name):
                    continue

                vec1[name] = random_direction(parameter.detach())
                vec2[name] = random_direction(parameter.detach())
            vectors.append((vec1, vec2))

        save_data(vectors_path, vectors)
        print(f'Created {len(vectors)} perturbation vector pairs.')

    # TEMPORARY: use CPU (GPU RAM limitations)
    if args.small_gpu:
        for pair in vectors:
            for v in pair:
                for name in v:
                    v[name] = v[name].cpu()

    # keep only those perturbation values specified in args.layer
    if args.layer != 'all':
        for pair in vectors:
            for v in pair:
                for name in v.keys() ^ {args.layer}:
                    v.pop(name, None)

    # enforce single kernel if necessary
    if args.single_kernel:
        for pair in vectors:
            for v in pair:
                for name in v:
                    z = torch.zeros_like(v[name])
                    z[0, 0] = v[name][0, 0]  # (nodes, channels, width, height)
                    v[name] = z

    gradients = gather_gradients(model, train_loader, criterion)
    learning_rate = 0.001
    #  step_size = args.step_scale * learning_rate * average_grad_std(gradients)
    step_size = args.max_step / ((args.grid_width-1) / 2)
    print(f'step size = {step_size}')

    # all_grids is a matrix of loss landscapes:
    #       batch1, batch2, batch3, ...
    # pair1   x       x       x     ...
    # pair2   x       x       x     ...
    # pair3   x       x       x     ...
    # ...
    print(f'Computing grid of {args.model_type} (w/ {args.activation}) on {args.dataset}')
    all_grids = []
    for pair_no, pair in enumerate(vectors):
        model_parameters = [params for name, params in model.named_parameters() if name in pair[0]]
        pair_parameters = [list(vec.values()) for vec in pair]
        #import ipdb; ipdb.set_trace()
        grid_walk = GridWalk(model_parameters, pair_parameters, step_size, args.grid_width)

        sample_grid(model, train_loader, criterion, grid_walk, desc=f"Perturbation {pair_no+1}/{len(vectors)}")
        grids = np.array(grid_walk.grid)

        # reorder axes from (v1, v2, batch) to (batch, v1, v2)
        grids = np.moveaxis(grids, -1, 0)
        grids = list(grids)  # batch-wise should be normal python list
        all_grids.append(grids)

    # grid data layout: (grid of grids, pair of permutation vectors, step size, extra info)
    save_data(grids_path, (all_grids, vectors, step_size, gradients))
    print(all_grids)

    # import ipdb; ipdb.set_trace()

if __name__ == '__main__':
    set_seed(args.seed)  # to get reproducible results
    main()
