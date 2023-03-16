from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.utils.model_zoo as model_zoo
import torch
import random
import numpy as np
import numpy.random as npr
__all__ = ['ResNet', 'resnet50', 'resnet18', 'resnet34', 'resnet101', 'resnet152']
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth'
}

def conv1x1(input_channel, output_channel,bias=False):
    return nn.Conv2d(input_channel, output_channel, kernel_size=1, bias=bias)


def conv3x3(in_channel, out_channel, stride=1, padding=1, bias=False):
    return nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=padding,bias=bias)

def random_sample(prob, sampling_num):
    batch_size, channels, h, w = prob.shape
    return torch.multinomial((prob.view(batch_size * channels, -1) + 1e-8), sampling_num, replacement=True)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, input_channel, output_channel, stride=1, downsample=None, track_running_stats=True):
        super(BasicBlock, self).__init__() 
        self.conv1 = conv3x3(input_channel, output_channel, stride=stride)
        self.bn1 = nn.BatchNorm2d(output_channel, track_running_stats=track_running_stats)
        self.relu = nn.ReLU(inplace=False) 
        self.conv2 = conv3x3(output_channel, output_channel)
        self.bn2 = nn.BatchNorm2d(output_channel, track_running_stats=track_running_stats)
        self.downsample = downsample
        self.stride = stride

    def forward(self, input):
        residual = input 
        out = self.conv1(input)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(input)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4 
    def __init__(self, input_channel, channel, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(input_channel, channel)
        self.bn1 = nn.BatchNorm2d(channel)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(channel, channel, stride=stride)
        self.bn2 = nn.BatchNorm2d(channel)
        self.conv3 = conv1x1(channel, channel*4)
        self.bn3 = nn.BatchNorm2d(channel*4)
        self.downsample = downsample
        self.stride = stride

    def forward(self, input):
        residual = input # skip path
        out = self.conv1(input)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(input)

        out += residual

        return self.relu(out)


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, backbone=False, feature_norm=False):
        self.feature_norm = feature_norm
        self.backbone = backbone
        super(ResNet, self).__init__()
        self.input_channel = 64 
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)      
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(kernel_size=7, padding=0, stride=1)
        self.fc_class = nn.Linear(512 * block.expansion, num_classes)
        self.baselayer = [self.conv1, self.bn1, self.layer1, self.layer2, self.layer3, self.layer4, self.fc_class]
        for m in self.modules():            
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1]*m.out_channels
                m.weight.data.normal_(0, math.sqrt(2./n))

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                

    def _make_layer(self, block, channel, blocks, stride=1):
        downsample = None
        if stride != 1 or self.input_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.input_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion),
            )

        layers = []
        layers.append(block(self.input_channel, channel, stride=stride, downsample=downsample))
        self.input_channel = channel*block.expansion
        for i in range(1, blocks):
            layers.append(block(self.input_channel, channel))

        return nn.Sequential(*layers)

    def forward(self, x):
        if x.size(-1) == 224:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.layer1(x)
        elif x.size(-1) == 56:
            x = self.layer1(x)
            
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        feature_out = x
            
        if self.backbone:
            return self.fc_class(x), feature_out
        else:
            x = self.fc_class(x)
            return x


class ResNet_RSC(nn.Module):
    def __init__(self, block, layers, num_classes=100):
        self.inplanes = 64
        super(ResNet_RSC, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        # self.jigsaw_classifier = nn.Linear(512 * block.expansion, jigsaw_classes)
        self.class_classifier = nn.Linear(512 * block.expansion, num_classes)
        #self.domain_classifier = nn.Linear(512 * block.expansion, domains)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def is_patch_based(self):
        return False

    def forward(self, x, gt=None, flag=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if flag: 
            self.eval()
            x_new = x.clone().detach()
            x_new = Variable(x_new.data, requires_grad=True) 
            x_new_view = self.avgpool(x_new)
            x_new_view = x_new_view.view(x_new_view.size(0), -1)
            # x_new_view = self.classifier(x_new_view)
            output = self.class_classifier(x_new_view)
            class_num = output.shape[1]
            index = gt
            num_rois = x_new.shape[0] # batch_size
            num_channel = x_new.shape[1]
            H = x_new.shape[2]
            HW = x_new.shape[2] * x_new.shape[3]
            one_hot = torch.zeros((1), dtype=torch.float32).cuda()
            one_hot = Variable(one_hot, requires_grad=False)
            sp_i = torch.ones([2, num_rois]).long()
            sp_i[0, :] = torch.arange(num_rois)
            sp_i[1, :] = index
            sp_v = torch.ones([num_rois])
            one_hot_sparse = torch.sparse.FloatTensor(sp_i, sp_v, torch.Size([num_rois, class_num])).to_dense().cuda()
            one_hot_sparse = Variable(one_hot_sparse, requires_grad=False)
            one_hot = torch.sum(output * one_hot_sparse)
            self.zero_grad()
            one_hot.backward()
            grads_val = x_new.grad.clone().detach() 
            grad_channel_mean = torch.mean(grads_val.view(num_rois, num_channel, -1), dim=2) 
            channel_mean = grad_channel_mean
            spatial_mean = torch.mean(grads_val, dim=1)
            spatial_mean = spatial_mean.view(num_rois, H, H).view(num_rois, HW)
            self.zero_grad()

            choose_one = random.randint(0, 9)
            if choose_one <= 4:
                # ---------------------------- spatial -----------------------
                spatial_drop_num = int(HW * 1/3)
                th_mask_value = torch.sort(spatial_mean, dim=1, descending=True)[0][:, spatial_drop_num]
                th_mask_value = th_mask_value.view(num_rois, 1).expand(num_rois, HW)
                mask_all_cuda = torch.where(spatial_mean >= th_mask_value, torch.zeros(spatial_mean.shape).cuda(),
                                            torch.ones(spatial_mean.shape).cuda())
                mask_all = mask_all_cuda.detach().cpu().numpy()
                for q in range(num_rois):
                    mask_all_temp = np.ones((HW), dtype=np.float32)
                    zero_index = np.where(mask_all[q, :] == 0)[0]
                    num_zero_index = zero_index.size
                    if num_zero_index >= spatial_drop_num:
                        dumy_index = npr.choice(zero_index, size=spatial_drop_num, replace=False)
                    else:
                        zero_index = np.arange(HW)
                        dumy_index = npr.choice(zero_index, size=spatial_drop_num, replace=False)
                    mask_all_temp[dumy_index] = 0
                    mask_all[q, :] = mask_all_temp
                mask_all = torch.from_numpy(mask_all.reshape(num_rois, 7, 7)).cuda()
                mask_all = mask_all.view(num_rois, 1, 7, 7)
            else:
                # -------------------------- channel ----------------------------
                mask_all = torch.zeros((num_rois, num_channel, 1, 1)).cuda()
                vector_thresh_percent = int(num_channel * 1 / 3.1)
                vector_thresh_value = torch.sort(channel_mean, dim=1, descending=True)[0][:, vector_thresh_percent]
                vector_thresh_value = vector_thresh_value.view(num_rois, 1).expand(num_rois, num_channel)
                vector = torch.where(channel_mean > vector_thresh_value,
                                     torch.zeros(channel_mean.shape).cuda(),
                                     torch.ones(channel_mean.shape).cuda())
                vector_all = vector.detach().cpu().numpy()
                channel_drop_num = int(num_channel * 1 / 3.2)
                vector_all_new = np.ones((num_rois, num_channel), dtype=np.float32)
                for q in range(num_rois):
                    vector_all_temp = np.ones((num_channel), dtype=np.float32)
                    zero_index = np.where(vector_all[q, :] == 0)[0]
                    num_zero_index = zero_index.size
                    if num_zero_index >= channel_drop_num:
                        dumy_index = npr.choice(zero_index, size=channel_drop_num, replace=False)
                    else:
                        zero_index = np.arange(num_channel)
                        dumy_index = npr.choice(zero_index, size=channel_drop_num, replace=False)
                    vector_all_temp[dumy_index] = 0
                    vector_all_new[q, :] = vector_all_temp
                vector = torch.from_numpy(vector_all_new).cuda()
                for m in range(num_rois):
                    index_channel = vector[m, :].nonzero()[:, 0].long()
                    index_channel = index_channel.detach().cpu().numpy().tolist()
                    mask_all[m, index_channel, :, :] = 1

            # ----------------------------------- batch ----------------------------------------
            cls_prob_before = F.softmax(output, dim=1)
            x_new_view_after = x_new * mask_all
            x_new_view_after = self.avgpool(x_new_view_after)
            x_new_view_after = x_new_view_after.view(x_new_view_after.size(0), -1)
            x_new_view_after = self.class_classifier(x_new_view_after)
            cls_prob_after = F.softmax(x_new_view_after, dim=1)

            sp_i = torch.ones([2, num_rois]).long()
            sp_i[0, :] = torch.arange(num_rois)
            sp_i[1, :] = index
            sp_v = torch.ones([num_rois])
            one_hot_sparse = torch.sparse.FloatTensor(sp_i, sp_v, torch.Size([num_rois, class_num])).to_dense().cuda()
            before_vector = torch.sum(one_hot_sparse * cls_prob_before, dim=1)
            after_vector = torch.sum(one_hot_sparse * cls_prob_after, dim=1)
            change_vector = before_vector - after_vector - 0.0001 
            change_vector = torch.where(change_vector > 0, change_vector, torch.zeros(change_vector.shape).cuda())
            th_fg_value = torch.sort(change_vector, dim=0, descending=True)[0][int(round(float(num_rois) * 1/3))]
            drop_index_fg = change_vector.gt(th_fg_value)
            ignore_index_fg = ~drop_index_fg 
            not_01_ignore_index_fg = ignore_index_fg.nonzero()[:, 0] 
            mask_all[not_01_ignore_index_fg.long(), :] = 1

            self.train()
            mask_all = Variable(mask_all, requires_grad=True)
            x = x * mask_all

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.class_classifier(x)


def resnet18(pretrained=True, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)
    return model

def resnet18_rsc(pretrained=True, **kwargs):
    model = ResNet_RSC(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)
    return model

def resnet50(pretrained=True, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict=False)
    return model

def resnet50_rsc(pretrained=True, **kwargs):
    model = ResNet_RSC(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict=False)
    return model
