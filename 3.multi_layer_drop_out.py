"""
多层感知机 + dropout
"""
import torch
import OM_reader
from torch import nn
from d2l import torch as d2l
from matplotlib import pyplot as plt
from OM_train import OM_train_device,OM_predict

if __name__ == "__main__":
    batch_size, lr, num_epochs = 64, 0.1, 100
    dropout1, dropout2 = 0.2, 0.5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 256),
        nn.ReLU(),
        # 在第一个全连接层之后添加一个dropout层
        nn.Dropout(dropout1),
        nn.Linear(256, 256),
        nn.ReLU(),
        # 在第二个全连接层之后添加一个dropout层
        nn.Dropout(dropout2),
        nn.Linear(256, 10)
    )

    train_loader,test_loader = OM_reader.load_oracle_mnist_data(batch_size)
    OM_train_device(net, train_loader, test_loader, num_epochs, lr, device)
    OM_predict(net, test_loader,device)
    plt.show()