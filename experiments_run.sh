#!/bin/bash
# resnet on cifar-10 (width=41 -> 25min per grid.py call)
python src/scripts/train.py resnet cifar-10;
python src/scripts/grid.py grid41e0 resnet cifar-10 --grid_width=41 --episode=0;
python src/scripts/grid.py grid41e1 resnet cifar-10 --grid_width=41 --episode=1;
python src/scripts/grid.py grid41e9 resnet cifar-10 --grid_width=41 --episode=9;
python src/scripts/grid.py grid41e1xs resnet cifar-10 --grid_width=41 --episode=1 --max_step=0.01;
python src/scripts/grid.py grid41e9xs resnet cifar-10 --grid_width=41 --episode=9 --max_step=0.01;

# sigmoid activation
python src/scripts/train.py resnet cifar-10 --activation=sigmoid;
python src/scripts/grid.py grid41e0 resnet cifar-10 --grid_width=41 --episode=0 --activation=sigmoid;
python src/scripts/grid.py grid41e1 resnet cifar-10 --grid_width=41 --episode=1 --activation=sigmoid;
python src/scripts/grid.py grid41e9 resnet cifar-10 --grid_width=41 --episode=9 --activation=sigmoid;

# tanh activation
python src/scripts/train.py resnet cifar-10 --activation=tanh;
python src/scripts/grid.py grid41e0 resnet cifar-10 --grid_width=41 --episode=0 --activation=tanh;
python src/scripts/grid.py grid41e1 resnet cifar-10 --grid_width=41 --episode=1 --activation=tanh;
python src/scripts/grid.py grid41e9 resnet cifar-10 --grid_width=41 --episode=9 --activation=tanh;

# fashion-mnist transfer (only resnet)
python src/scripts/train.py resnet fashion-mnist;
python src/scripts/grid.py grid41e0 resnet fashion-mnist --grid_width=41 --episode=0;
python src/scripts/grid.py grid41e1 resnet fashion-mnist --grid_width=41 --episode=1;
python src/scripts/grid.py grid41e9 resnet fashion-mnist --grid_width=41 --episode=9;

# vgg on cifar-10 (width=41 -> 100min per grid.py call)
python src/scripts/train.py vgg cifar-10;
python src/scripts/grid.py grid41e1 vgg cifar-10 --small_gpu --grid_width=41 --episode=1;
python src/scripts/grid.py grid41e9 vgg cifar-10 --small_gpu --grid_width=41 --episode=9;
python src/scripts/grid.py grid41e1xs vgg cifar-10 --small_gpu --grid_width=41 --episode=1 --max_step=0.01;
python src/scripts/grid.py grid41e9xs vgg cifar-10 --small_gpu --grid_width=41 --episode=9 --max_step=0.01;


