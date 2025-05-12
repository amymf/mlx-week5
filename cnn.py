import torch.nn as nn
import torch.nn.functional as F

class UrbanSoundCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(UrbanSoundCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(128 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # x shape (batch_size, 1, 128, 128)
        x = self.pool(F.relu(self.conv1(x)))
        # x shape (batch_size, 32, 64, 64)
        x = self.pool(F.relu(self.conv2(x)))
        # x shape (batch_size, 64, 32, 32)
        x = self.pool(F.relu(self.conv3(x)))
        # x shape (batch_size, 128, 16, 16)
        x = x.view(-1, 128 * 16 * 16)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x