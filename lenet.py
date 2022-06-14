"""
lenet(CNN)
"""
import torch
from torch import nn
from d2l import torch as d2l
import OM_reader
from matplotlib import pyplot as plt
from multi_layer_perceptrons import OM_train_GPU,OM_predict

if __name__ == "__main__":
    batch_size, lr, num_epochs = 64, 0.1, 60
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = nn.Sequential(
        nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(6, 16, kernel_size=5), nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(16 * 5 * 5 , 120), nn.ReLU(),
        nn.Linear(120, 84), nn.ReLU(),
        nn.Linear(84, 10)
    )

    trainer = torch.optim.SGD(net.parameters(), lr=lr)

    train_loader,test_loader = OM_reader.load_oracle_mnist_data(batch_size)
    OM_train_GPU(net, train_loader, test_loader, num_epochs, lr, device)
    OM_predict(net, test_loader,device)
    plt.show()