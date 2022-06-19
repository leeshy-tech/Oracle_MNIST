"""
多层感知机
"""
import torch
import OM_reader
from torch import nn
from matplotlib import pyplot as plt
from OM_show import get_oracle_mnist_labels,show_images
from OM_train import OM_train_device,OM_predict

if __name__ == "__main__":
    batch_size, lr, num_epochs = 64, 0.1, 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = nn.Sequential(nn.Flatten(),
                        nn.Linear(784, 256),
                        nn.ReLU(),
                        nn.Linear(256, 10))

    train_loader,test_loader = OM_reader.load_oracle_mnist_data(batch_size)
    OM_train_device(net, train_loader, test_loader, num_epochs, lr, device)
    OM_predict(net, test_loader,device)
    plt.show()