from torch import nn


class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1,
                                             padding='same'), nn.BatchNorm2d(num_features=8), nn.LeakyReLU(0.01))
        self.pool1 = nn.Sequential(nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=8, out_channels=32, kernel_size=3, stride=1,
                                             padding='same'),  nn.BatchNorm2d(num_features=32), nn.LeakyReLU(0.01))
        self.pool2 = nn.Sequential(nn.MaxPool2d(2))
        self.fc1 = nn.Sequential(nn.Linear(in_features=1568, out_features=512), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(in_features=512, out_features=2048), nn.ReLU())
        self.fc3 = nn.Sequential(nn.Linear(in_features=2048, out_features=512), nn.ReLU())
        self.fc4 = nn.Linear(in_features=512, out_features=num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)

        # flatten
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x


class Personal(nn.Module):
    def __init__(self, num_classes=10):
        super(Personal, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1,
                                             padding='same'), nn.BatchNorm2d(num_features=8), nn.LeakyReLU(0.01))
        self.pool1 = nn.Sequential(nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=8, out_channels=32, kernel_size=3, stride=1,
                                             padding='same'), nn.BatchNorm2d(num_features=32), nn.LeakyReLU(0.01))
        self.pool2 = nn.Sequential(nn.MaxPool2d(2))
        self.fc1 = nn.Sequential(nn.Linear(in_features=1568, out_features=512), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(in_features=512, out_features=2048), nn.ReLU())
        self.fc3 = nn.Sequential(nn.Linear(in_features=2048, out_features=512), nn.ReLU())
        self.pl1 = nn.Sequential(nn.Linear(in_features=512, out_features=2048), nn.ReLU())
        self.pl2 = nn.Sequential(nn.Linear(in_features=2048, out_features=512), nn.ReLU())
        self.fc4 = nn.Linear(in_features=512, out_features=num_classes)


def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)

        # flatten
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.pl1(x)
        x = self.pl2(x)
        x = self.fc4(x)
        return x
