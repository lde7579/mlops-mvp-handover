"""소형 CNN: 128x128 흑백 이미지 1장 -> NG 로짓 1개 (sigmoid 를 씌우면 NG 확률)."""
from torch import nn


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()

        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))

        self.features = nn.Sequential(block(1, 8), block(8, 16), block(16, 32), block(32, 32))  # 128 -> 8
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(32 * 8 * 8, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        return self.head(self.features(x)).squeeze(1)  # 로짓
