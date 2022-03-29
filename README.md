# landscapes

Exploring loss landscapes

## Installation
[anaconda](https://www.anaconda.com/products/individual) simplifies dependency management. To install, execute:
```bash
conda env create -f torch-land.yml
conda activate torch-land
export PYTHONPATH=.
```

## Scripts

All scripts are run from the projects root directory and further specify their usage when called with the `-h` flag.

### Training
Training a model whose loss landscapes we want to investigate later:
```bash
python src/scripts/train.py resnet fashion-mnist
```

### Computing loss landscapes
After having trained a model, to compute loss landscapes (= losses over a 2-dimensional parameter subspace) use the `gird.py` script, e.g.
```bash
python src/scripts/grid.py grid9 resnet fashion-mnist --grid_width=9
```

### Visualizing the results
Now, that the loss values have been computed, visualize the landscapes using 2d heatmaps by calling the `visualize.py` script with the same parameters.
```bash
python src/scripts/visualize.py grid9 resnet fashion-mnist --grid_width=9
```

## Experiments
The commands to run the experiments are documented in the files [experiments_run.sh](experiments_run.sh) and [experiments_visualize.sh](experiments_visualize.sh).

The landscapes are computed using a pair of random filter-normalized vectors that perturb the model's parameters. The losses correspond to a training step, meaning only a single mini-batch.

We use three pairs of perturbation vectors and the training-set's first three mini-batches of 256 images.

For visualization, we can either look at heatmaps or contour-plots (using the `--contour` flag on `visualize.py`):
<p align="middle">
  <img src="images/cifar_resnet/grid41e0.png" width="400" />
  <img src="images/cifar_resnet/grid41e0_heatmap.png" width="400" />
</p>

### Some results

Training progress: ResNet on CIFAR-10 before training and after the first and ninth episode
<p align="middle">
  <img src="images/cifar_resnet/grid41e0.png" width="400" />
  <img src="images/cifar_resnet/grid41e1.png" width="400" />
  <img src="images/cifar_resnet/grid41e9.png" width="400" />
</p>
Training progress on Fashion-MNIST
<p align="middle">
  <img src="images/fashion_resnet/grid41e0.png" width="400" />
  <img src="images/fashion_resnet/grid41e1.png" width="400" />
  <img src="images/fashion_resnet/grid41e9.png" width="400" />
</p>

Different activation functions before and after overfitting: ReLU, sigmoid and tanh
<p align="middle">
  <img src="images/cifar_resnet/grid41e1.png" width="400" />
  <img src="images/cifar_resnet/grid41e9.png" width="400" />
</p>
<p align="middle">
  <img src="images/cifar_resnet_sigmoid/grid41e1.png" width="400" />
  <img src="images/cifar_resnet_sigmoid/grid41e9.png" width="400" />
</p>
<p align="middle">
  <img src="images/cifar_resnet_tanh/grid41e1.png" width="400" />
  <img src="images/cifar_resnet_tanh/grid41e9.png" width="400" />
</p>

Network architecture: Resnet14 vs VGG11
<p align="middle">
  <img src="images/cifar_resnet/grid41e1.png" width="400" />
  <img src="images/cifar_vgg/grid41e1.png" width="400" />
</p>
<p align="middle">
  <img src="images/cifar_resnet/grid41e9.png" width="400" />
  <img src="images/cifar_vgg/grid41e9.png" width="400" />
</p>

Zooming in 100x (VGG one shows only noise)
<p align="middle">
  <img src="images/cifar_resnet/grid41e9xs.png" width="400" />
  <img src="images/cifar_vgg/grid41e9xs.png" width="400" />
</p>