import torch
from torch import nn
import OM_reader
from matplotlib import pyplot as plt
from OM_train import OM_train_device,OM_predict

if __name__ == "__main__":
    batch_size,lr, num_epochs = 64, 0.1, 30
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = nn.Sequential(
        # 这里，我们使用一个11*11的更大窗口来捕捉对象。
        # 同时，步幅为4，以减少输出的高度和宽度。
        # 另外，输出通道的数目远大于LeNet
        nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2),
        # 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数
        nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2),
        # 使用三个连续的卷积层和较小的卷积窗口。
        # 除了最后的卷积层，输出通道的数量进一步增加。
        # 在前两个卷积层之后，汇聚层不用于减少输入的高度和宽度
        nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
        nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
        nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.Flatten(),
        # 这里，全连接层的输出数量是LeNet中的好几倍。使用dropout层来减轻过拟合
        nn.Linear(6400, 4096), nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(4096, 4096), nn.ReLU(),
        nn.Dropout(p=0.5),
        # 最后是输出层。由于这里使用Oracle-MNIST，所以用类别数为10，而非论文中的1000
        nn.Linear(4096, 10))

    train_loader, test_loader = OM_reader.load_oracle_mnist_data(batch_size, resize=224)
    OM_train_device(net, train_loader, test_loader, num_epochs, lr, device,ylim=[0,1])
    OM_predict(net, test_loader,device,size=224)
    plt.show()