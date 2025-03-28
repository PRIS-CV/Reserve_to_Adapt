import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd.variable import *
from torchvision import models
import os
import numpy as np
from utilities import *
from typing import Union


class BaseFeatureExtractor(nn.Module):
    def forward(self, *input):
        pass

    def __init__(self):
        super(BaseFeatureExtractor, self).__init__()
    def output_num(self):
        pass

resnet_dict = {"resnet18":models.resnet18, "resnet34":models.resnet34, "resnet50":models.resnet50, "resnet101":models.resnet101, "resnet152":models.resnet152}

class ResNetFc(BaseFeatureExtractor):
    def __init__(self, model_name='resnet50',model_path=None, normalize=True):
        super(ResNetFc, self).__init__()
        self.model_resnet = resnet_dict[model_name](pretrained=False)
        if not os.path.exists(model_path):
            model_path = None
            print('invalid model path!')
        if model_path:
            self.model_resnet.load_state_dict(torch.load(model_path))
        if model_path or normalize:
            self.normalize = True
            self.mean = False
            self.std = False
        else:
            self.normalize = False

        model_resnet = self.model_resnet
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.__in_features = model_resnet.fc.in_features

    def get_mean(self):
        if self.mean is False:
            self.mean = Variable(
                torch.from_numpy(np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape((1, 3, 1, 1)))).cuda()
        return self.mean

    def get_std(self):
        if self.std is False:
            self.std = Variable(
                torch.from_numpy(np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape((1, 3, 1, 1)))).cuda()
        return self.std

    def forward(self, x):
        if self.normalize:
            x = (x - self.get_mean()) / self.get_std()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

    def output_num(self):
        return self.__in_features

class CLS(nn.Module):
    def __init__(self, in_dim, out_dim, bottle_neck_dim=256,  temp=0.05):
        super(CLS, self).__init__()
        self.temp = 1#nn.Parameter(torch.ones(1,device='cuda'), requires_grad=True)
        if bottle_neck_dim:
            self.bottleneck = nn.Linear(in_dim, bottle_neck_dim)
            self.weight1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad = True)
            self.fc = nn.Linear(bottle_neck_dim, out_dim, bias = False)
            
            self.main = nn.Sequential(
                self.bottleneck,
                nn.Sequential(
                    nn.BatchNorm1d(bottle_neck_dim),
                    nn.LeakyReLU(0.2, inplace=True),
                    self.fc
                ),
                nn.Softmax(dim=-1)
            )
        else:
            self.fc = nn.Linear(in_dim, out_dim)
            # if fc_init is not None:
            #     nn.init.constant_(self.fc.weight, fc_init)
            self.main = nn.Sequential(
                self.fc,
                nn.Softmax(dim=-1)
            )

    def forward(self, x):
        out = [x]
        
        for i, module in enumerate(self.main.children()):
            if i==0:
                x = module(x)
                x = x/torch.norm(x, dim =-1,keepdim=True)
            else:
                x = module(x)
            out.append(x)
       
        out[-2] = out[-2]/ self.temp
        out[-1] = nn.Softmax(dim=-1)(out[-2])
        return out
    
    def virt_forward(self, K, feature_source, logits: torch.Tensor, target: Union[torch.Tensor, None] = None,  ) -> torch.Tensor:
        if self.training:
            with torch.no_grad():
                W_yi = torch.gather(self.fc.weight, 0, target.unsqueeze(1).expand(target.size(0), self.fc.weight.size(1)))   
                W_virt = torch.norm(W_yi,dim=1).unsqueeze(-1).unsqueeze(-1) * ((K / torch.norm(K, dim =1).unsqueeze(-1)).unsqueeze(0))
            vir = torch.bmm(W_virt, feature_source.unsqueeze(-1)).squeeze(-1)
            logits = torch.cat([logits, vir], dim=-1)
            x = nn.Softmax(-1)(logits)
        return x
    

class AdversarialNetwork(nn.Module):
    def __init__(self):
        super(AdversarialNetwork, self).__init__()
        self.main = nn.Sequential()
        self.grl = GradientReverseModule(lambda step: aToBSheduler(step, 0.0, 1.0, gamma=10, max_iter=10000))

    def forward(self, x):
        x = self.grl(x)
        for module in self.main.children():
            x = module(x)
        return x

class LargeAdversarialNetwork(AdversarialNetwork):
    def __init__(self, in_feature):
        super(LargeAdversarialNetwork, self).__init__()
        self.ad_layer1 = nn.Linear(in_feature, 1024)
        self.ad_layer2 = nn.Linear(1024, 1024)
        self.ad_layer3 = nn.Linear(1024, 1)
        self.sigmoid = nn.Sigmoid()

        self.main = nn.Sequential(
            self.ad_layer1,
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            self.ad_layer2,
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            self.ad_layer3,
            self.sigmoid
        )


class GradientReverseLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, coeff, input):
        ctx.coeff = coeff
        return input

    @staticmethod
    def backward(ctx, grad_outputs):
        coeff = ctx.coeff
        return None, -coeff * grad_outputs
    
class GradientReverseModule(nn.Module):
    def __init__(self, scheduler):
        super(GradientReverseModule, self).__init__()
        self.scheduler = scheduler
        self.global_step = 0.0
        self.coeff = 0.0
        self.grl = GradientReverseLayer.apply
    def forward(self, x):
        self.coeff = self.scheduler(self.global_step)
        self.global_step += 1.0
        return self.grl(self.coeff, x)
