from torch import nn

class ResLayer(nn.Module):
    def __init__(self, in_channels, t, c, s):
        super(ResLayer, self).__init__()
        self.in_channels = in_channels
        self.c = c
        self.s = s

        latent_channels = in_channels * t
        self.bottle = nn.Sequential(
            nn.Conv2d(in_channels, latent_channels, 1, bias=False),
            nn.BatchNorm2d(latent_channels),
            nn.LeakyReLU(),

            nn.Conv2d(latent_channels, latent_channels, 3, stride=s, padding=1, groups=latent_channels, bias=False),
            nn.BatchNorm2d(latent_channels),
            nn.LeakyReLU(),

            nn.Conv2d(latent_channels, c, 1, bias=False),
            nn.BatchNorm2d(c)
        )

    def forward(self, x):
        bottle = self.bottle(x)

        if self.in_channels == self.c and self.s == 1:
            return bottle + x
        else:
            return bottle

class FrontNet(nn.Module):
    def __init__(self, width_mult=1, **kw):
        super(FrontNet, self).__init__()
        self.bottlenecks = [
            [0, 32, 2],
            [1, 16, 1],

            [6, 24, 2],
            [6, 24, 1],

            [6, 32, 2]
        ]
        layers = []
        in_channels = 3
        for t, c, s in self.bottlenecks:
            out_channels = int(c * width_mult)
            if t == 0:
                layers.append(self.conv2d(in_channels, out_channels, s))
            else:
                layers.append(ResLayer(in_channels, t, out_channels, s))
            in_channels = out_channels

        self.layers = nn.Sequential(*layers)

    def conv2d(self, inp, outp, stride):
        kernel = 1 if stride == 1 else 3
        padding = (kernel-1)//2
        return nn.Sequential(
            nn.Conv2d(inp, outp, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(outp),
            nn.ReLU6(True)
        )

    def forward(self, x):
        return self.layers(x)

class MiddleNet(nn.Module):
    def __init__(self, width_mult=1, **kw):
        super(MiddleNet, self).__init__()
        self.bottlenecks = [
            [6, 32, 1],
            [6, 32, 1],
            [6, 64, 2]
        ]
        layers = []
        in_channels = 32
        for t, c, s in self.bottlenecks:
            out_channels = int(c * width_mult)
            layers.append(ResLayer(in_channels, t, out_channels, s))
            in_channels = out_channels

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class CloudNet(nn.Module):
    def __init__(self, num_classes, width_mult=1, **kw):
        super(CloudNet, self).__init__()
        self.bottlenecks = [
            [6, 64, 1],
            [6, 64, 1],
            [6, 64, 1],

            [6, 96, 1],
            [6, 96, 1],
            [6, 96, 1],

            [6, 160, 2],
            [6, 160, 1],
            [6, 160, 1],

            [6, 320, 1],
            [0, 400, 1]
        ]
        layers = []
        in_channels = 64
        for t, c, s in self.bottlenecks:
            out_channels = int(c * width_mult)
            if t == 0:
                layers.append(self.conv2d(in_channels, out_channels, s))
            else:
                layers.append(ResLayer(in_channels, t, out_channels, s))
            in_channels = out_channels

        self.layers = nn.Sequential(*layers)
        self.avgpool7 = nn.AvgPool2d(7)
        self.line = nn.Sequential(
            nn.Dropout(),
            nn.Linear(in_channels, num_classes)
        )

    def conv2d(self, inp, outp, stride):
        kernel = 1 if stride == 1 else 3
        padding = (kernel-1)//2
        return nn.Sequential(
            nn.Conv2d(inp, outp, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(outp),
            nn.ReLU6(True)
        )

    def forward(self, x):
        x = self.layers(x)
        x = self.avgpool7(x)
        x = x.view(x.size()[0], -1)
        return self.line(x)
