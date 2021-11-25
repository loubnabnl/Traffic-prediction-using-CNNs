
import torch.nn as nn
""" we want a traffic prediction model to predict one month traffic 
mor multiple (location, direction) couples based on the past 3/5 months data

We use a CNN with a sliding window on the Time Series data"""

class TimeCNN(nn.Module):
    def __init__(self):
        super(TimeCNN, self).__init__()
        #1st convolutional layer
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        #2nd convolutional layer
        self.layer2 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(24*14)
        )
        #fully connected layers
        self.fc1 = nn.Linear(in_features=64*24*14, out_features=120)
        self.drop = nn.Dropout2d(0.3)
        self.fc2 = nn.Linear(in_features=120, out_features=24*30)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.drop(out)
        out = self.fc2(out)
        return out