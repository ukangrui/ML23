from __future__ import print_function
from PIL import Image
import numpy as np
import torch
import torch.utils.data as data

class NumpyImageDataset(data.Dataset):

    # image dataset from numpy arrays

    # Example:

    # num_classes = 20
    # train_data = np.load('../data/cifar_train_data.npy').transpose((0,2,3,1))
    # train_label = np.load('../data/cifar_train_label.npy')
    # trainset = NumpyImageDataset(train_data, train_label, transform=transform_train)
    # trainloader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)    

    def __init__(self, data_array, label_array, transform=None, label_transform=None):
        self.data_tensor = data_array
        self.label_tensor = torch.from_numpy(label_array).type('torch.LongTensor')
        self.transform = transform
        self.label_transform = label_transform
        assert self.data_tensor.shape[0] == self.label_tensor.size(0)

    def __getitem__(self, index):

        img, label = self.data_tensor[index], self.label_tensor[index]

        # change to PIL image
        img = Image.fromarray(img)

        if self.transform is not None:
            # apply user-defined sequence of transformations
            img = self.transform(img)

        if self.label_transform is not None:
            label = self.label_transform(label)

        return img, label

    def __len__(self):
        return self.data_tensor.shape[0]
    
