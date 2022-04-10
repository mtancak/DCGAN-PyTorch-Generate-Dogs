import torch
import torch.nn as nn
from enum import Enum


class GAN_MODE(Enum):
    DISCRIMINATOR = 1
    GENERATOR = 2


class DCGAN(nn.Module):
    def __init__(self, shape, in_channels, out_channels, mode_limit):
        super(DCGAN, self).__init__()

        self.shape = shape
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.discriminator = nn.Sequential()
        self.generator = nn.Sequential()

        self.mode = GAN_MODE.DISCRIMINATOR
        self.mode_executions = 0
        self.mode_limit = mode_limit

    def forward(self, x):
        pass


if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(DEVICE)

    model = DCGAN((64, 64), 3, 1, 10).to(DEVICE)
    x = torch.rand(1, 3, 64, 64).to(DEVICE)
    # y = torch.rand(1, 1, 64, 64).to(DEVICE)
    #print("output shape x = " + str(.shape))

    model(x)
