import torch.nn as nn
import torch.nn.functional as F

class Simple_Net(nn.Module):
    def __init__(self):
        super(Simple_Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 200)
        self.fc2 = nn.Linear(200, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        feature = self.fc1(x)
        x = self.fc2(F.relu(feature))
        return feature, F.log_softmax(x, dim=1)