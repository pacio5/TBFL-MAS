from torch import nn

# CNN model
class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1,
                                             padding='same'), nn.BatchNorm2d(num_features=64), nn.LeakyReLU(0.01))
        self.pool1 = nn.Sequential(nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1,
                                             padding='same'), nn.BatchNorm2d(num_features=128), nn.LeakyReLU(0.01))
        self.pool2 = nn.Sequential(nn.MaxPool2d(2))
        self.fc1 = nn.Sequential(nn.Linear(in_features=6272, out_features=1028), nn.ReLU())
        self.fc2 = nn.Linear(in_features=1028, out_features=num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)

        # flatten
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

# class for FedPER
class PersonalCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(PersonalCNN, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1,
                                             padding='same'), nn.BatchNorm2d(num_features=64), nn.LeakyReLU(0.01))
        self.pool1 = nn.Sequential(nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1,
                                             padding='same'), nn.BatchNorm2d(num_features=128), nn.LeakyReLU(0.01))
        self.pool2 = nn.Sequential(nn.MaxPool2d(2))
        self.fc1 = nn.Sequential(nn.Linear(in_features=6272, out_features=1028), nn.ReLU())

        # personal layers
        self.pl1 = nn.Sequential(nn.Linear(in_features=1028, out_features=4096), nn.ReLU())
        self.pl2 = nn.Sequential(nn.Linear(in_features=4096, out_features=1028), nn.ReLU())

        self.fc2 = nn.Linear(in_features=1028, out_features=num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)

        # flatten
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.pl1(x)
        x = self.pl2(x)
        x = self.fc2(x)
        return x
