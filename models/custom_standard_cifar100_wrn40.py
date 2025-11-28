import math
import torch
import torch.nn as nn
import torch.nn.functional as F






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





class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0, use_buffer = False, alpha_init=1e-5):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None

        self.use_buffer = use_buffer
        
        if self.use_buffer:
            self.buffer1 = BufferLayer(in_planes)
            self.alpha_buffer1 = nn.Parameter(torch.tensor(alpha_init))

            self.buffer2 = BufferLayer(out_planes)
            self.alpha_buffer2 = nn.Parameter(torch.tensor(alpha_init))
 




    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
            if self.use_buffer:
                x = self.buffer1(x) * self.alpha_buffer1 + x
        else:
            out = self.relu1(self.bn1(x))
            if self.use_buffer:
                out = self.buffer1(out) * self.alpha_buffer1 + out


        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))

        if self.use_buffer:
            out = self.buffer2(out) * self.alpha_buffer2 + out

        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0, use_buffer= False, alpha_init=1e-5):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate, use_buffer, alpha_init)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate, use_buffer= False, alpha_init=1e-5):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate, use_buffer, alpha_init))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    """ Based on code from https://github.com/yaodongyu/TRADES """
    # def __init__(self, depth=28, num_classes=10, widen_factor=10, sub_block1=False, dropRate=0.0, bias_last=True):
    def __init__(self, depth=34, num_classes=10, widen_factor=10, sub_block1=True, dropRate=0.0, bias_last=True, use_buffers= [True, False, False], alpha_init=1e-5):
        super(WideResNet, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate, use_buffer=use_buffers[0], alpha_init=alpha_init)
        if sub_block1:
            self.sub_block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate, use_buffer=use_buffers[1], alpha_init=alpha_init)
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate, use_buffer=use_buffers[2], alpha_init=alpha_init)


        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes, bias=bias_last)
        self.nChannels = nChannels[3]



        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear) and not m.bias is None:
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)



class WideResNet_TTA(WideResNet):

    def __init__(self, depth=40, widen_factor=2, use_buffers= [False, False, False], alpha_init=1e-5):
        super().__init__(depth=depth,
                         widen_factor=widen_factor,
                         sub_block1=False,
                         num_classes=100,
                         use_buffers=use_buffers,
                         alpha_init=alpha_init)
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