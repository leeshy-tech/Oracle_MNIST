import os
import gzip
import parameters
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
class ImageList(Dataset):

    def __init__(self, path, kind, transform=None):
        (train_set, train_labels) = load_data(path, kind)
        self.train_set = train_set
        self.train_labels = train_labels
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.train_set[index], int(self.train_labels[index])
        if self.transform is not None:
            img = self.transform(np.array(img))
        return img, target

    def __len__(self):
        return len(self.train_set)

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

def load_oracle_mnist_data(batch_size):
    train_data = ImageList(path=parameters.path, kind='train',
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.5,), (0.5,))
                           ]))

    test_data = ImageList(path=parameters.path, kind='t10k',
                          transform=transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,))
                          ]))

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader,test_loader