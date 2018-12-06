# README

## Population Based Training (PBT) method 
Original paper by [DeepMind](https://arxiv.org/abs/1711.09846).
Here PBT is applied to a simple example of training an mnist classifier.

### Prerequisite
```
pytorch 0.3
python3
scipy
argparse
tensorboard-pytorch
tensorboardX
path.py
evo
```

### Project structure

#### Config file
Config file structure is described in config/config.md.

In general, dataset, model, train version, test versions and, optionally, metric and loss, 
are automatically initialized based on the definition in a config file.

#### Main
To run a vanilla (without PBT) training and testing (as parallel threads), from the project root directory 
```
python3 main.py path/to/config/file
```
The `-m train` option allows to run only train, `-m test` to run only test and `-m traintest` to run 
train followed by a test.

To run a PBT training (testing will be run every several epochs, as defined in the config file) 
```
python3 main_PBT.py path/to/config/file
```
