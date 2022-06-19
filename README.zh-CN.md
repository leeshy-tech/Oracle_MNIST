# Oracle_MNIST
[![Readme-EN](https://img.shields.io/badge/README-English-green.svg)](README.md)

> 本repo在2022/6/20 0:00 由私有仓库转为公开仓库

利用数据集“Oracle-MNIST”进行的一些深度学习实践。

## 数据集 Oracle-MNIST
数据集“Oracle-MNIST”来自于BUPT邓伟洪老师团队

可以直接到工程[https://github.com/wm-bupt/oracle-mnist](https://github.com/wm-bupt/oracle-mnist)下查看。

## 代码说明
### 结构
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
└─ oracle_minist_data
```
- oracle_minist_data：Oracle-MNIST数据集
- parameters.py：一些参数
- OM_reader.py：读取数据集的帮助类、函数
- OM_show.py：可视化数据集中的图像
- OM_train.py：训练模型函数
- 1.linear_neural_network.py：线性神经网络
- 2.multi_layer_perceptrons.py：多层感知机
- 3.multi_layer_drop_out.py：多层感知机+dropout
- 4.LeNet.py：LeNet
- 5.AlexNet.py：AlexNet
- 6.ResNet.py：ResNet
- 7.DenseNet.py：DenseNet
## 参考
[https://github.com/wm-bupt/oracle-mnist](https://github.com/wm-bupt/oracle-mnist)

[https://github.com/d2l-ai/d2l-zh](https://github.com/d2l-ai/d2l-zh)