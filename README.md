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
Training a model, whose loss landscapes we want to investigate later:
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
