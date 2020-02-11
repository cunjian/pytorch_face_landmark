'''AlexNet for CIFAR10. FC layers are removed. Paddings are adjusted.
Without BN, the start learning rate should be 0.01
(c) YANG, Wei
'''
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ConvBlock(nn.Module):
    def __init__(self, inp, oup, k, s, p, dw=False, linear=False):
        super(ConvBlock, self).__init__()
        self.linear = linear
        if dw:
            self.conv = nn.Conv2d(inp, oup, k, s, p, groups=inp, bias=False)
        else:
            self.conv = nn.Conv2d(inp, oup, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(oup)
        if not linear:
            self.prelu = nn.PReLU(oup)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.linear:
            return x
        else:
            return self.prelu(x)
            
class ResNet(nn.Module):
    def __init__(self,num_classes):
        super(ResNet,self).__init__()
        self.pretrain_net = models.resnet18(pretrained=False)
        self.base_net = nn.Sequential(*list(self.pretrain_net.children())[:-1])
        self.fc = nn.Linear(512,num_classes) # resnet18:512; resnet50:2048
    def forward(self,x):
        x = self.base_net(x)
        #print(x.size()) #(256, 512, 1, 1)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

        

# USE global depthwise convolution layer.
class MobileNet_GDConv(nn.Module):
    def __init__(self,num_classes):
        super(MobileNet_GDConv,self).__init__()
        self.pretrain_net = models.mobilenet_v2(pretrained=False)
        self.base_net = nn.Sequential(*list(self.pretrain_net.children())[:-1])
        #self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        #self.lastconv = nn.Conv2d(1280, out_channels=num_classes, kernel_size=1)
        self.linear7 = ConvBlock(1280, 1280, (7, 7), 1, 0, dw=True, linear=True)
        self.linear1 = ConvBlock(1280, num_classes, 1, 1, 0, linear=True)
        #self.fc = nn.Linear(1280,num_classes) # resnet18:512; resnet50:2048
    def forward(self,x):
        x = self.base_net(x)
        #print(x.shape) # [64, 1280, 7, 7]
        #x = self.avgpool(x)
        #x = self.lastconv(x)
        x = self.linear7(x)
        x = self.linear1(x)
        x = x.view(x.size(0), -1)
        #x = self.fc(x)
        return x

# USE global depthwise convolution layer.
class MobileNet_GDConv_56(nn.Module):
    def __init__(self,num_classes):
        super(MobileNet_GDConv_56,self).__init__()
        self.pretrain_net = models.mobilenet_v2(pretrained=False)
        self.base_net = nn.Sequential(*list(self.pretrain_net.children())[:-1])
        #self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        #self.lastconv = nn.Conv2d(1280, out_channels=num_classes, kernel_size=1)
        self.linear7 = ConvBlock(1280, 1280, (2, 2), 1, 0, dw=True, linear=True)
        self.linear1 = ConvBlock(1280, num_classes, 1, 1, 0, linear=True)
        #self.fc = nn.Linear(1280,num_classes) # resnet18:512; resnet50:2048
    def forward(self,x):
        x = self.base_net(x)
        #print(x.shape) # [64, 1280, 7, 7]
        #x = self.avgpool(x)
        #x = self.lastconv(x)
        x = self.linear7(x)
        x = self.linear1(x)
        x = x.view(x.size(0), -1)
        #x = self.fc(x)
        return x        
        