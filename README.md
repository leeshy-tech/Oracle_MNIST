# Oracle_MNIST
[![Readme-CN](https://img.shields.io/badge/README-中文-green.svg)](README.zh-CN.md)

> this repo is changed from private to public on 2022/6/20 0:00

Some simple deep learning practices on dataset "Oracle-MNIST"
## Dataset Oracle-MNIST
The dataset "Oracle-MNIST" comes from the team of Mr. Deng Weihong from BUPT.

You can view it directly under the project [https://github.com/wm-bupt/oracle-mnist](https://github.com/wm-bupt/oracle-mnist).

## code description
### Structure
```
Oracle-MNIST
├─ OM_reader.py
├─ OM_show.py
├─ OM_train.py
├─ 1.linear_neural_network.py
├─ 2.multi_layer_perceptrons.py
├─ 3.multi_layer_drop_out.py
├─ 4.LeNet.py
├─ 5.AlexNet.py
├─ 6.ResNet.py
├─ 7.DenseNet.py
├─ parameters.py
├─ README.md
├─ README.zh-CN.md
└─ oracle_minist_data
```
- oracle_minist_data: Oracle-MNIST dataset
- parameters.py: some parameters
- OM_reader.py: helper classes and functions for reading datasets
- OM_show.py: Visualize the images in the dataset
- OM_train.py：functions to train the model
- 1.linear_neural_network.py：linear neural network
- 2.multi_layer_perceptrons.py：multi layer perceptrons
- 3.multi_layer_drop_out.py：multi layer perceptrons + dropout
- 4.LeNet.py：LeNet
- 5.AlexNet.py：AlexNet
- 6.ResNet.py：ResNet
- 7.DenseNet.py：DenseNet
## reference
[https://github.com/wm-bupt/oracle-mnist](https://github.com/wm-bupt/oracle-mnist)

[https://github.com/d2l-ai/d2l-zh](https://github.com/d2l-ai/d2l-zh)