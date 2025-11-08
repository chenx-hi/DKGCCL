import torch.nn as nn


class LogReg(nn.Module):
    def __init__(self, hidden_channels, num_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(hidden_channels, num_classes)

    def forward(self, x):
        out = self.fc(x)
        return out
