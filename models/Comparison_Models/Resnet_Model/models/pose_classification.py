import torch
import torch.nn as nn
import torch.nn.functional as F


class TennisPoseClassifier(nn.Module):
    # input: raw image tensor of shape (B, 3, H, W)
    # output: logits of shape (B, 4)

    def __init__(self, num_classes: int = 4, dropout: float = 0.3):
        super().__init__()

        # 8 convolution layers
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        # halves H and W each time it is applied
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(512 * 16 * 16, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # Block 1
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool(x)
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = self.pool(x)

        # make output size fixed
        x = torch.flatten(x, 1)

        # Dense layers
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)

        # raw logits for CrossEntropyLoss
        return x