import torch
import torch.nn as nn


class SquatClassifierCNN(nn.Module):
    def __init__(self, input_dim, dropout_rate=0.25):
        """
        Args:
            input_dim: tuple (frames, features) e.g., (30, 66)
        """
        super().__init__()
        frames, features = input_dim
        
        # Calculate output sizes after each layer
        # Input: (batch, 1, 30, 66)
        
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # (batch, 32, 30, 66)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),       # (batch, 32, 15, 33)
            nn.Dropout2d(dropout_rate)
        )
        
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # (batch, 64, 15, 33)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),       # (batch, 64, 7, 16)
            nn.Dropout2d(dropout_rate)
        )
        
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),# (batch, 128, 7, 16)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),                # (batch, 128, 1, 1)
            nn.Dropout2d(dropout_rate)
        )
        
        # Classifier (input is 128 from adaptive pooling)
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # x shape: (batch, 1, 30, 66)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)  # (batch, 128, 1, 1)
        x = x.view(x.size(0), -1)  # Flatten: (batch, 128)
        x = self.classifier(x)
        return x
