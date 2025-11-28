import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import init




class BufferLayer(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.conv1 = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        
        self.scale1 = torch.tensor(0.5)
        self.scale2 = torch.tensor(0.5)
        

    def forward(self, x):
        

        out = self.scale1 * self.conv1(x) + self.scale2 * self.conv3(x)
        

        return out




class ResNeXtBottleneck(nn.Module):
    """
    ResNeXt Bottleneck Block type C (https://github.com/facebookresearch/ResNeXt/blob/master/models/resnext.lua).
    """
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 cardinality,
                 base_width,
                 stride=1,
                 downsample=None,
                 use_buffer = False,
                 alpha_init=1e-5):
        super(ResNeXtBottleneck, self).__init__()

        dim = int(math.floor(planes * (base_width / 64.0)))
        
        self.use_buffer= use_buffer

        self.conv_reduce = nn.Conv2d(
            inplanes,
            dim * cardinality,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False)
        self.bn_reduce = nn.BatchNorm2d(dim * cardinality)

        self.conv_conv = nn.Conv2d(
            dim * cardinality,
            dim * cardinality,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=cardinality,
            bias=False)
        self.bn = nn.BatchNorm2d(dim * cardinality)

        self.conv_expand = nn.Conv2d(
            dim * cardinality,
            planes * 4,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False)
        self.bn_expand = nn.BatchNorm2d(planes * 4)

        self.downsample = downsample


        
        if self.use_buffer:
            self.buffer1 = BufferLayer(dim * cardinality)
            self.alpha_buffer1 = nn.Parameter(torch.tensor(alpha_init))
            
            self.buffer2 = BufferLayer(dim * cardinality)
            self.alpha_buffer2 = nn.Parameter(torch.tensor(alpha_init))
 


    def forward(self, x):
        residual = x
        
        
        bottleneck = self.conv_reduce(x)
        

        bottleneck = self.bn_reduce(bottleneck)
        
        bottleneck = F.relu(bottleneck, inplace=True)
        if self.use_buffer:
            bottleneck = self.buffer1(bottleneck) * self.alpha_buffer1 + bottleneck
        
        
        bottleneck = self.conv_conv(bottleneck)
        
        bottleneck = self.bn(bottleneck)

        bottleneck = F.relu(bottleneck, inplace=True)
        if self.use_buffer:
            bottleneck = self.buffer2(bottleneck) * self.alpha_buffer2 + bottleneck

        
        bottleneck = self.conv_expand(bottleneck)
        bottleneck = self.bn_expand(bottleneck)
        

        if self.downsample is not None:
            residual = self.downsample(x)

        returns = F.relu(residual + bottleneck, inplace=True)
        


        return returns




class CifarResNeXt(nn.Module):
    """ResNext optimized for the Cifar dataset, as specified in
    https://arxiv.org/pdf/1611.05431.pdf."""

    def __init__(self, block, depth, cardinality, base_width, num_classes, use_buffers, alpha_init):
        super(CifarResNeXt, self).__init__()

        # Model type specifies number of layers for CIFAR-10 and CIFAR-100 model
        assert (depth - 2) % 9 == 0, 'depth should be one of 29, 38, 47, 56, 101'
        layer_blocks = (depth - 2) // 9

        self.cardinality = cardinality
        self.base_width = base_width
        self.num_classes = num_classes

        self.conv_1_3x3 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)

        self.buffer0 = BufferLayer(64)
        self.alpha_buffer0 = nn.Parameter(torch.tensor(alpha_init))
        self.bn_1 = nn.BatchNorm2d(64)


        self.inplanes = 64
        self.stage_1 = self._make_layer(block, 64, layer_blocks,  1, use_buffers[0], alpha_init)
        self.stage_2 = self._make_layer(block, 128, layer_blocks, 2, use_buffers[1], alpha_init)
        self.stage_3 = self._make_layer(block, 256, layer_blocks, 2, use_buffers[2], alpha_init)

        self.avgpool = nn.AvgPool2d(8)
        self.classifier = nn.Linear(256 * block.expansion, num_classes)



        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight)
                m.bias.data.zero_()



    def _make_layer(self, block, planes, blocks, stride=1, use_buffer= False, alpha_init=1e-5):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False),
                nn.BatchNorm2d(planes * block.expansion))


        layers = []
        layers.append(block(self.inplanes, planes, self.cardinality, self.base_width, stride,downsample = downsample, use_buffer = use_buffer, alpha_init = alpha_init))
        self.inplanes = planes * block.expansion
        
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, self.cardinality, self.base_width, use_buffer = use_buffer, alpha_init = alpha_init))


        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_1_3x3(x)
        
        

        x = self.bn_1(x)
        x = F.relu(x, inplace=True)
        x = self.buffer0(x) * self.alpha_buffer0 + x
        
        x = self.stage_1(x)
        x = self.stage_2(x)
        x = self.stage_3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)







class Hendrycks2020AugMixResNeXtNet(CifarResNeXt):

    def __init__(self, depth=29, cardinality=4, base_width=32, use_buffers= [False, False, False], alpha_init=1e-5):
        super().__init__(ResNeXtBottleneck,
                         depth=depth,
                         num_classes=100,
                         cardinality=cardinality,
                         base_width=base_width,
                         use_buffers=use_buffers,
                         alpha_init=alpha_init
                         )
        self.register_buffer('mu', torch.tensor([0.5] * 3).view(1, 3, 1, 1))
        self.register_buffer('sigma', torch.tensor([0.5] * 3).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mu) / self.sigma
        return super().forward(x)











def load_partial_weights(model, weight_path):
    old_weights = torch.load(weight_path, map_location=torch.device('cpu'))

    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in old_weights.items() if k in model_dict and model_dict[k].shape == v.shape}

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict, strict=False)
    print(f"Loaded {len(pretrained_dict.keys())} layers from the pretrained model.")

def load_partial_weights_from_model(target_model, source_model):
    source_dict = source_model.state_dict()
    target_dict = target_model.state_dict()

    # 키가 일치하고, shape이 동일한 파라미터만 가져오기
    pretrained_dict = {k: v for k, v in source_dict.items() if k in target_dict and target_dict[k].shape == v.shape}

    target_dict.update(pretrained_dict)
    target_model.load_state_dict(target_dict, strict=False)
    print(f"Loaded {len(pretrained_dict.keys())} layers from the source model.")

## batch 256 best