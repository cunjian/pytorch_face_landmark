'''AlexNet for CIFAR10. FC layers are removed. Paddings are adjusted.
Without BN, the start learning rate should be 0.01
(c) YANG, Wei
'''
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


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




class MobileNet(nn.Module):
    def __init__(self,num_classes):
        super(MobileNet,self).__init__()
        self.pretrain_net = models.mobilenet_v2(pretrained=False)
        self.base_net = nn.Sequential(*list(self.pretrain_net.children())[:-1])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1280,num_classes) # resnet18:512; resnet50:2048
    def forward(self,x):
        x = self.base_net(x)
        #print(x.size()) #(256, 512, 1, 1)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class MobileNet_FCN(nn.Module):
    def __init__(self,num_classes):
        super(MobileNet_FCN,self).__init__()
        self.pretrain_net = models.mobilenet_v2(pretrained=False)
        self.base_net = nn.Sequential(*list(self.pretrain_net.children())[:-1])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.lastconv = nn.Conv2d(1280, out_channels=num_classes, kernel_size=1)
        #self.fc = nn.Linear(1280,num_classes) # resnet18:512; resnet50:2048
    def forward(self,x):
        x = self.base_net(x)
        #print(x.size()) #(256, 512, 1, 1)
        x = self.avgpool(x)
        x = self.lastconv(x)
        x = x.view(x.size(0), -1)
        #x = self.fc(x)
        return x
        
