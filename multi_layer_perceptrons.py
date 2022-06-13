"""
多层感知机
"""
import torch
import OM_reader
from torch import nn
from d2l import torch as d2l
from matplotlib import pyplot as plt
from OM_show import get_oracle_mnist_labels,show_images

def OM_train_GPU(net, train_iter, test_iter, num_epochs, lr, device):
    """用GPU训练模型"""
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])
    timer, num_batches = d2l.Timer(), len(train_iter)
    for epoch in range(num_epochs):
        # 训练损失之和，训练准确率之和，样本数
        metric = d2l.Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, train_acc, None))
            #plt.pause(0.1)

        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')

def OM_predict(net, test_iter,device, n=7):
    """Predict labels"""
    for X, y in test_iter:
        break
    trues = get_oracle_mnist_labels(y)
    preds = get_oracle_mnist_labels(d2l.argmax(net(X.to(device)).to(device), axis=1))
    titles = [true +'\n' + pred for true, pred in zip(trues, preds)]
    show_images(d2l.reshape(X[0:n], (n, 28, 28)), 1, n, titles=titles[0:n])

if __name__ == "__main__":
    batch_size, lr, num_epochs = 256, 0.1, 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = nn.Sequential(nn.Flatten(),
                        nn.Linear(784, 256),
                        nn.ReLU(),
                        nn.Linear(256, 10))

    net.apply(init_weights)
    loss = nn.CrossEntropyLoss(reduction='none')
    trainer = torch.optim.SGD(net.parameters(), lr=lr)

    train_loader,test_loader = OM_reader.load_oracle_mnist_data(batch_size)
    OM_train_GPU(net, train_loader, test_loader, num_epochs, lr, device)
    OM_predict(net, test_loader,device)
    plt.show()