import os
import gzip
import parameters
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset,DataLoader
from torchvision import datasets, transforms
class OracleMNIST(Dataset):
    def __init__(self, path, kind, transform=None):
        (data, labels) = load_data(path, kind)
        self.data = data
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, labels = self.data[index], int(self.labels[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img, mode='L')
        if self.transform is not None:
            img = self.transform(img)

        return img, labels

    def __len__(self) -> int:
        return len(self.data)

def load_data(path, kind='train'):
    """Load Oracle-MNIST data from `path`"""
    labels_path = os.path.join(path, '%s-labels-idx1-ubyte.gz' % kind)
    images_path = os.path.join(path, '%s-images-idx3-ubyte.gz' % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 28, 28)

    print('The size of %s set: %d'%(kind, len(labels)))

    return images, labels

def load_oracle_mnist_data(batch_size,resize=None):
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = OracleMNIST(
        path="./oracle_minist_data", kind='train', transform=trans)
    mnist_test = OracleMNIST(
        path="./oracle_minist_data", kind='t10k', transform=trans)
    return (DataLoader(mnist_train, batch_size, shuffle=True),
            DataLoader(mnist_test, batch_size, shuffle=False)
            )

train_loader,test_loader = load_oracle_mnist_data(18)
