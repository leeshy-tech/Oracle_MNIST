"""
线性神经网络
"""
import parameters
import torch
import OM_reader
from torch import nn
from matplotlib import pyplot as plt
from OM_show import show_images,get_oracle_mnist_labels
from d2l import torch as d2l
from OM_train import OM_train_device,OM_predict

def OM_train(net, train_iter, test_iter, loss, num_epochs, updater):
    """Train a model"""
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss/4', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_loss, train_acc = d2l.train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = d2l.evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, (train_loss/4 , train_acc) + (test_acc,))
    
    return train_loss,train_acc,test_acc

"""def OM_predict(net, test_iter, n=7):
    
    for X, y in test_iter:
        break
    trues = get_oracle_mnist_labels(y)
    preds = get_oracle_mnist_labels(d2l.argmax(net(X), axis=1))
    titles = [true +'\n' + pred for true, pred in zip(trues, preds)]
    show_images(d2l.reshape(X[0:n], (n, 28, 28)), 1, n, titles=titles[0:n])
"""
if __name__ == "__main__":
    # 用CPU训练就足够了
    device = torch.device("cpu")
    batch_size, lr, num_epochs = 256, 0.1, 30

    train_loader,test_loader = OM_reader.load_oracle_mnist_data(batch_size)

    net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))

    OM_train_device(net, train_loader, test_loader, num_epochs,lr,device)
    OM_predict(net, test_loader, device)
    plt.show()