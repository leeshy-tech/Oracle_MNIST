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

def OM_train(net, train_iter, test_iter, loss, num_epochs, updater):
    """Train a model"""
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss/4', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_loss, train_acc = d2l.train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = d2l.evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, (train_loss/4 , train_acc) + (test_acc,))
    
    return train_loss,train_acc,test_acc
    """assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc"""

def OM_predict(net, test_iter, n=7):
    """Predict labels"""
    for X, y in test_iter:
        break
    trues = get_oracle_mnist_labels(y)
    preds = get_oracle_mnist_labels(d2l.argmax(net(X), axis=1))
    titles = [true +'\n' + pred for true, pred in zip(trues, preds)]
    show_images(d2l.reshape(X[0:n], (n, 28, 28)), 1, n, titles=titles[0:n])

if __name__ == "__main__":
    batch_size = 1024
    num_epochs = 30
    train_loader,test_loader = OM_reader.load_oracle_mnist_data(batch_size)
    net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))

    loss = nn.CrossEntropyLoss(reduction='none')
    trainer = torch.optim.SGD(net.parameters(), lr=0.1)

    train_loss,train_acc,test_acc = OM_train(net, train_loader, test_loader, loss, num_epochs, trainer)
    print("train_loss:" + str(train_loss))
    print("train_acc:" + str(train_acc))
    print("test_acc:" + str(test_acc))
    OM_predict(net, test_loader)
    plt.show()