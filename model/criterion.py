from torch import nn

class Criterion(nn.Module):
    def __init__(self):
        super(Criterion, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, y):
        return self.criterion(x, y)
