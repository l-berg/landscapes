#!/bin/bash
# resnet on cifar-10
python src/scripts/visualize.py grid41e0 resnet cifar-10 --grid_width=41 --episode=0 --contour;
python src/scripts/visualize.py grid41e1 resnet cifar-10 --grid_width=41 --episode=1 --contour;
python src/scripts/visualize.py grid41e9 resnet cifar-10 --grid_width=41 --episode=9 --contour;
python src/scripts/visualize.py grid41e1xs resnet cifar-10 --grid_width=41 --episode=1 --max_step=0.01 --contour;
python src/scripts/visualize.py grid41e9xs resnet cifar-10 --grid_width=41 --episode=9 --max_step=0.01 --contour;

# sigmoid activation
python src/scripts/visualize.py grid41e0 resnet cifar-10 --grid_width=41 --episode=0 --activation=sigmoid --contour;
python src/scripts/visualize.py grid41e1 resnet cifar-10 --grid_width=41 --episode=1 --activation=sigmoid --contour;
python src/scripts/visualize.py grid41e9 resnet cifar-10 --grid_width=41 --episode=9 --activation=sigmoid --contour;

# tanh activation
python src/scripts/visualize.py grid41e0 resnet cifar-10 --grid_width=41 --episode=0 --activation=tanh --contour;
python src/scripts/visualize.py grid41e1 resnet cifar-10 --grid_width=41 --episode=1 --activation=tanh --contour;
python src/scripts/visualize.py grid41e9 resnet cifar-10 --grid_width=41 --episode=9 --activation=tanh --contour;

# fashion-mnist transfer (only resnet)
python src/scripts/visualize.py grid41e0 resnet fashion-mnist --grid_width=41 --episode=0 --contour;
python src/scripts/visualize.py grid41e1 resnet fashion-mnist --grid_width=41 --episode=1 --contour;
python src/scripts/visualize.py grid41e9 resnet fashion-mnist --grid_width=41 --episode=9 --contour;

# vgg on cifar-10
python src/scripts/visualize.py grid41e1 vgg cifar-10 --small_gpu --grid_width=41 --episode=1 --clip=3 --contour;
python src/scripts/visualize.py grid41e9 vgg cifar-10 --small_gpu --grid_width=41 --episode=9 --clip=3 --contour;
python src/scripts/visualize.py grid41e1xs vgg cifar-10 --small_gpu --grid_width=41 --episode=1 --max_step=0.01 --contour;
python src/scripts/visualize.py grid41e9xs vgg cifar-10 --small_gpu --grid_width=41 --episode=9 --max_step=0.01 --contour;


