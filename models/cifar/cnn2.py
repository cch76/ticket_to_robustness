from torch.nn import Module
from torch import nn
import torch.nn.functional as F
import torch

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, 1)
        self.conv2 = nn.Conv2d(64, 64, 3, 1)

        # self.conv3 = nn.Conv2d(64, 128, 3, 1)
        # self.conv4 = nn.Conv2d(128, 128, 3, 1)

        # self.conv5 = nn.Conv2d(64, 128, 3, 1)
        # self.conv6 = nn.Conv2d(128, 256, 3, 1)

        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        # x = self.conv3(x)
        # x = F.relu(x)
        # x = self.conv4(x)
        # x = F.relu(x)
        # x = F.max_pool2d(x, 2)

        # x = self.conv5(x)
        # x = F.relu(x)
        # x = self.conv6(x)
        # x = F.relu(x)
        # x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

def cnn():
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    """
    model = Net()
    return model