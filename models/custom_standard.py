import math
import torch
import torch.nn as nn
import torch.nn.functional as F





class BufferLayer(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.conv1 = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        
        self.scale1 = nn.Parameter(torch.tensor(0.5))
        self.scale2 = nn.Parameter(torch.tensor(0.5))

        

    def forward(self, x):
        

        out = self.scale1 * self.conv1(x) + self.scale2 * self.conv3(x)
        

        return out





class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0, use_buffer=False, alpha_init=1e-5):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False) or None

        self.use_buffer = use_buffer

        if self.use_buffer :
        
            self.buffer_1 = BufferLayer(in_planes)
            self.alpha_buffer_1 = nn.Parameter(torch.tensor(alpha_init))

            self.buffer_2 = BufferLayer(out_planes)
            self.alpha_buffer_2 = nn.Parameter(torch.tensor(alpha_init))
# 


    def forward(self, x):

        if not self.equalInOut:
            afterbn = self.bn1(x)
            x = self.relu1(afterbn)
        else:
            afterbn = self.bn1(x)
            out = self.relu1(afterbn)
        inter = out if self.equalInOut else x

        if  self.use_buffer :
            inter = self.buffer_1(inter) * self.alpha_buffer_1 + inter
            
        afterconv   = self.conv1(inter)

        afterbn     = self.bn2(afterconv)
        
        afterrelu   = self.relu2(afterbn)

        if  self.use_buffer :
            afterrelu = self.buffer_2(afterrelu) * self.alpha_buffer_2 + afterrelu

        if self.droprate > 0:
            afterrelu = F.dropout(afterrelu, p=self.droprate, training=self.training)

        out = self.conv2(afterrelu)
        
        returns = torch.add(x if self.equalInOut else self.convShortcut(x), out)


        return returns






class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0, use_buffer = False, alpha_init=1e-5):
        super(NetworkBlock, self).__init__()
        self.layer = nn.ModuleList()
        for i in range(nb_layers):
            self.layer.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate, use_buffer, alpha_init))

    def forward(self, x):
        for layer in self.layer:
            x = layer(x)
        return x





class WideResNetTTA(nn.Module):
    def __init__(self, depth=28, num_classes=10, widen_factor=10, dropRate=0.0, bias_last=True, use_buffers = [False, False, False], alpha_init=1e-5):
        super(WideResNetTTA, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) // 6
        block = BasicBlock


        self.alpha_init = alpha_init
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False)
        
        ############ initial buffer layer ############
        self.buffer_0 = BufferLayer(nChannels[0])
        self.alpha_buffer_0 = nn.Parameter(torch.tensor(self.alpha_init))
        ############ initial buffer layer ############

        
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate, use_buffers[0], self.alpha_init)
        
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate, use_buffers[1], self.alpha_init)
        
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate, use_buffers[2], self.alpha_init)


        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes, bias=bias_last)
        self.nChannels = nChannels[3]

    def forward(self, x):
        out = self.conv1(x)

        out = self.buffer_0(out) * self.alpha_buffer_0 + out

        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)

        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)








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

    



def load_partial_weights_from_model_exclude_prefix(target_model: torch.nn.Module, source_model: torch.nn.Module):
    """
    source_model.state_dict() 에 'model.' prefix 가 있으면 제거한 뒤,
    이름과 shape 이 일치하는 파라미터만 가져와 target_model 에 로드합니다.
    """
    # 1) 원본과 타겟의 state_dict
    source_dict = source_model.state_dict()
    target_dict = target_model.state_dict()

    # 2) prefix 자동 감지: 'model.' 이 붙어 있으면 제거
    prefix = ''
    for k in source_dict:
        if k.startswith('model.'):
            prefix = 'model.'
            break

    # 3) prefix 제거 및 매핑
    mapped = {}
    for k, v in source_dict.items():
        new_k = k[len(prefix):] if k.startswith(prefix) else k
        mapped[new_k] = v

    # 4) 이름·shape 일치하는 것만 필터링
    matched = {
        k: v
        for k, v in mapped.items()
        if k in target_dict and target_dict[k].shape == v.shape
    }

    # 5) 업데이트·로드
    target_dict.update(matched)
    target_model.load_state_dict(target_dict, strict=False)
    

    # target_model.bn_tta0.bn_tta.load_state_dict(target_model.bn1.state_dict())

    print(f"Loaded {len(matched)} / {len(target_dict)} layers from the source model.")