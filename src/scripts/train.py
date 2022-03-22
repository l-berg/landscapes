from src.utils.utils import set_seed, save_data, load_data, get_model, get_dataloaders, get_args_path

import torch
from torch import nn

import time
import os
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description='''
Train a model. Sets up directory structure with checkpoints, losses etc. for further processing.
''')

parser.add_argument('model_type', choices=['resnet', 'vgg'])
parser.add_argument('dataset', choices=['cifar-10', 'tiny-imagenet', 'fashion-mnist', 'mnist'])
parser.add_argument('--activation', default='relu', choices=['relu', 'sigmoid', 'tanh'],
                    help="make sure to keep this consistent between calls to train, grid and visualize")
parser.add_argument("--nepochs", type=int, default=10, help="number of epochs")
parser.add_argument("--batch_size", type=int, default=128, help="batch size")
parser.add_argument("--nworkers", type=int, default=4, help="number of workers for dataloaders")
parser.add_argument("--seed", type=int, default=0, help="random seed")
# parser.add_argument('--checkpoint_every', type=int, default=1000)
# parser.add_argument('--checkpoint')
# parser.add_argument('--load', dest='load', default=False, action='store_true')
args = parser.parse_args()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using {}'.format(device))


def run_model(net, loader, criterion, optimizer, train = True):
    running_loss = 0
    running_accuracy = 0

    # Set mode
    if train:
        net.train()
    else:
        net.eval()


    for i, (X, y) in enumerate(loader):
        # Pass to gpu or cpu
        X, y = X.to(device), y.to(device)

        # Zero the gradient
        optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            output = net(X)
            _, pred = torch.max(output, 1)
            loss = criterion(output, y)

        # If on train backpropagate
        if train:
            loss.backward()
            optimizer.step()

        # Calculate stats
        running_loss += loss.item()
        running_accuracy += torch.sum(pred == y.detach())
    return running_loss / len(loader), (running_accuracy.double() / len(loader.dataset)).item()

def main():
    args_path = get_args_path(args.dataset, args.model_type, args.activation)

    # unique directory for every call of the script
    def get_run(r):
        return os.path.join(args_path, f'training_run_{r}')
    run = 0
    while os.path.exists(get_run(run)):
        run += 1
    path = get_run(run)
    Path(path).mkdir(parents=True, exist_ok=True)

    # create common symlink
    symlink_path = os.path.join(args_path, 'training_run')
    symlink_path_tmp = f"{symlink_path}_tmp"
    os.symlink(os.path.basename(path), symlink_path_tmp)
    os.rename(symlink_path_tmp, symlink_path)

    train_loader, val_loader = get_dataloaders(args.dataset, batch_size=args.batch_size, nworkers=args.nworkers)
    _, in_channels, in_width, _ = next(iter(train_loader))[0].shape
    model = get_model(args.model_type, in_channels, in_width, activation=args.activation).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    stat_list = []
    print(f'\nStarting training of {args.model_type} (w/ {args.activation}) on {args.dataset}')
    print('0th epoch is without training!\n\n')
    for e in range(args.nepochs):
        start = time.time()
        train_loss, train_acc = run_model(model, train_loader, criterion, optimizer, train=(e > 0))
        val_loss, val_acc = run_model(model, val_loader, criterion, optimizer, train=False)
        end = time.time()

        stat_str = '\t'.join([f"Epoch: {e}", f"train loss: {train_loss:.3f}, train acc: {train_acc:.3f}",
                              f"val loss: {val_loss:.3f}, val acc: {val_acc:.3f}", f"time: {end-start:.1f}s"])
        print(stat_str)

        stats = {
            'episode': e,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'time': end-start,
        }
        stat_list.append(stats)

        # save checkpoint and stats in new subdirectory
        episode_path = os.path.join(path, str(e))
        Path(episode_path).mkdir()
        torch.save(model.state_dict(), os.path.join(episode_path, 'model.pt'))
        save_data(os.path.join(episode_path, 'stats'), stats)
        # also save everything in parent directory for ease of use
        save_data(os.path.join(path, 'stats'), stat_list)


if __name__ == '__main__':
    set_seed(args.seed, device)  # to get reproducible results
    main()