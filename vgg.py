import torch
import torch.nn as nn
from collapsible_mlp import CollapsibleMlp


class VGG(nn.Module):
    def __init__(self, vgg_name, num_classes=1000):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = CollapsibleMlp(2048 if num_classes == 200 else 512, 4096, num_classes, batch_norm=True)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif type(x) != int and x.split(' ')[0] == 'D':
                p = float(x.split(' ')[1])
                layers += [nn.Dropout(p=p)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1, bias=False),
                           nn.BatchNorm2d(x),
                           nn.PReLU(num_parameters=1, init=0.0)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

    def show_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.show_params()

    def get_slopes(self, fraction=1.0):
        num_mlp_layers = len(list(self.named_modules()))
        for name, module in list(self.named_modules())[::-1][:int(num_mlp_layers * fraction)]:
            if isinstance(module, CollapsibleMlp):
                print(name, module.act.weight.item())
            if isinstance(module, nn.PReLU):
                print(name, module.weight.item())

    def get_linear_loss(self, fraction=1.0):
        linear_loss = 0
        num_mlp_layers = len(list(self.named_modules()))
        for name, module in list(self.named_modules())[::-1][:int(num_mlp_layers * fraction)]:
            if isinstance(module, CollapsibleMlp):
                linear_loss += module.linear_loss()
            if isinstance(module, nn.PReLU):
                linear_loss += (1 - module.weight) ** 2
        return linear_loss


class VGG16(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG16, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.2))

        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.3))

        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.PReLU(num_parameters=1, init=0.1))
        self.layer7 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.PReLU(num_parameters=1, init=0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.4))

        self.layer8 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.PReLU(num_parameters=1, init=0.1))
        self.layer9 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.PReLU(num_parameters=1, init=0.1))
        self.layer10 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.PReLU(num_parameters=1, init=0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.4))

        self.layer11 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.PReLU(num_parameters=1, init=0.1))
        self.layer12 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.PReLU(num_parameters=1, init=0.1))
        self.layer13 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.PReLU(num_parameters=1, init=0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.5))
        self.fc = nn.Sequential(
            # nn.Linear(7*7*512, 4096),
            nn.Linear(512, 4096),
            nn.BatchNorm1d(4096),
            nn.PReLU(num_parameters=1, init=0.1))
        self.fc1 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.PReLU(num_parameters=1, init=0.1),
            nn.Dropout(0.5))
        self.fc2 = nn.Sequential(
            nn.Linear(4096, num_classes))
        # self.fc_small = nn.Sequential(
        #     nn.Dropout(0.5),
        #     # nn.Linear(7*7*512, num_classes))
        #     nn.Linear(512, num_classes))

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.layer10(out)
        out = self.layer11(out)
        out = self.layer12(out)
        out = self.layer13(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        # out = self.fc_small(out)
        return out