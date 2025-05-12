import torch.nn as nn
import torch.nn.functional as F

class UrbanSoundCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(UrbanSoundCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.dropout1 = nn.Dropout(0.3)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.dropout2 = nn.Dropout(0.3)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.dropout3 = nn.Dropout(0.3)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.fc1 = nn.Linear(128 * 16 * 16, 512)
        self.bn4 = nn.BatchNorm1d(512)
        self.dropout4 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # x shape (batch_size, 1, 128, 128)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # Conv1 -> BatchNorm -> ReLU -> Pool
        x = self.dropout1(x)
        # x shape (batch_size, 32, 64, 64)
        x = self.pool(F.relu(self.bn2(self.conv2(x)))) # Conv2 -> BatchNorm -> ReLU -> Pool
        x = self.dropout2(x)
        # x shape (batch_size, 64, 32, 32)
        x = self.pool(F.relu(self.bn3(self.conv3(x)))) # Conv3 -> BatchNorm -> ReLU -> Pool
        x = self.dropout3(x)
        # x shape (batch_size, 128, 16, 16)
        x = x.view(-1, 128 * 16 * 16)  # Flatten the tensor
        # x shape (batch_size, 32768)
        x = F.relu(self.bn4(self.fc1(x))) # FC1 -> BatchNorm -> ReLU
        x = self.dropout4(x)
        x = self.fc2(x)
        return x