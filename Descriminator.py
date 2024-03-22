import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)  # Output: 64x112x112
        self.conv1_1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)  # Output: 64x112x112
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)  # Output: 64x112x112

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # Output: 128x56x56
        self.conv2_1 = nn.Conv2d(128, 128, kernel_size=3, padding=1) # Output: 128x56x56
        self.conv2_1 = nn.Conv2d(128, 128, kernel_size=3, padding=1) # Output: 128x56x56
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1) # Output: 256x28x28
        self.conv3_1 = nn.Conv2d(256, 256, kernel_size=3, stride = 2, padding=1) # Output: 256x28x28
        self.conv3_1 = nn.Conv2d(256, 256, kernel_size=3, padding=1) # Output: 256x28x28
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.fc = nn.Linear(128*224*224, 1)

    def forward(self, x):
        x = F.gelu(self.bn1(self.conv1(x)))
        x = F.gelu(self.bn1(self.conv1_1(x)))
        x = F.gelu((self.conv1_2(x)))
        x = F.gelu(self.bn2(self.conv2(x)))
        x = F.gelu(self.bn2(self.conv2_1(x)))
        x = F.gelu((self.conv2_1(x)))
        # x = F.gelu(self.bn3(self.conv3(x)))
        # x = F.gelu(self.bn3(self.conv3_1(x)))
        # x = F.gelu(self.bn3(self.conv3_1(x)))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return torch.sigmoid(x)

