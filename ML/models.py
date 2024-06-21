import numpy as np
from sklearn.metrics import accuracy_score
from torch import nn
import torch
import torch.nn.functional as F
import dataset
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.transforms import Resize, ToTensor, Compose
from tqdm import tqdm
import os
import pandas as pd


class MNIST_Model(nn.Module):
    def __init__(self, num_classes=10):
        super(MNIST_Model, self).__init__()
        self.conv1 = self._make_block(1, 8, 3)
        self.conv2 = self._make_block(8, 32, 3)

        # flatten
        self.fc1 = nn.Sequential(nn.Linear(in_features=1568, out_features=512), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(in_features=512, out_features=1024), nn.ReLU())
        self.fc3 = nn.Sequential(nn.Linear(in_features=1024, out_features=2048), nn.ReLU())
        self.fc4 = nn.Sequential(nn.Linear(in_features=2048, out_features=512), nn.ReLU())
        self.fc5 = nn.Sequential(nn.Linear(in_features=512, out_features=num_classes))

    def _make_block(self, in_channels, out_channels, kernel_size):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1,
                      padding='same'),
            nn.BatchNorm2d(num_features=out_channels),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        # flatten
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        return x

class FashionMNIST(Dataset):
    def __init__(self, root, transform=None, is_train=True):
        if is_train:
            root = os.path.join(root, "fashion-mnist_train.csv")
        else:
            root = os.path.join(root, "fashion-mnist_test.csv")
        # read csv
        MNIST_data = pd.read_csv(root, sep=',')
        MNIST_data = np.array(MNIST_data, dtype='float32')
        # normalize
        x = MNIST_data[:, 1:]
        y = MNIST_data[:, 0]

        image_rows = 28
        image_cols = 28
        image_shape = (image_rows, image_cols, 1)
        x = x.reshape(x.shape[0], *image_shape)

        self.x = x
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, item):
        label = self.y[item]
        image = self.x[item]
        if self.transform:
            image = self.transform(image)
        return image, label



if __name__ == "__main__":
    data_path = '../fashion dataset'

    transforms = Compose([ToTensor()])

    train_dataset = FashionMNIST(root=data_path, transform=transforms, is_train=True)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=20,
                                  num_workers=6,
                                  shuffle=True,
                                  drop_last=True)
    test_dataset = FashionMNIST(root=data_path, transform=transforms, is_train=False)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=20,
                                 num_workers=6,
                                 shuffle=False,
                                 drop_last=True)

    model = MNIST_Model(10)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    best_acc = -1
    epochs = 1
    for epoch in range(epochs + 1):
        progress_bar = tqdm(train_dataloader, colour="cyan")
        model.train()
        for i, (images, labels) in enumerate(progress_bar):
            labels = labels.type(torch.LongTensor)
            images, labels = images.to(device), labels.to(device)
            print(labels.shape)
            print(images.shape)
            outputs = model(images)
            loss = criterion(outputs, labels)
            #print(model.fc1[0].weight.grad)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        all_losses = []
        all_labels = []
        all_predictions = []
        model.eval()

        with torch.no_grad():
            for i, (images, labels) in enumerate(test_dataloader):
                labels = labels.type(torch.LongTensor)
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)  # 20images 10 predicts
                losses = criterion(outputs, labels)
                predictions = torch.argmax(outputs, dim=1)  # take the prediction elements

                # add elements into list
                all_losses.append(losses.item())
                all_predictions.extend(predictions.tolist())
                all_labels.extend(labels.tolist())
            loss = np.mean(all_losses)
            accuracy = accuracy_score(all_labels, all_predictions)

            if accuracy > best_acc:  # save acc
                best_acc = accuracy

    print(best_acc)
    print(model.state_dict()['conv1.0.weight'])
