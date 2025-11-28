"""
Builds upon: https://github.com/DequanWang/tent
Corresponding paper: https://arxiv.org/abs/2006.10726
"""
import torch
import torch.nn as nn

from methods.base import TTAMethod
from utils.registry import ADAPTATION_REGISTRY
from utils.losses import Entropy


@ADAPTATION_REGISTRY.register()
class Tent_buffer(TTAMethod):
    """Tent adapts a model by entropy minimization during testing.
    Once tented, a model adapts itself by updating on every forward.
    """
    def __init__(self, cfg, model, num_classes):
        self.with_bn = cfg.WITH_BN
        super().__init__(cfg, model, num_classes)

        # setup loss function
        self.softmax_entropy = Entropy()
        

    def loss_calculation(self, x):
        imgs_test = x[0]
        outputs = self.model(imgs_test)
        loss = self.softmax_entropy(outputs).mean(0)
        return outputs, loss

    @torch.enable_grad()
    def forward_and_adapt(self, x):
        """Forward and adapt model on batch of data.
        Measure entropy of the model prediction, take gradients, and update params.
        """
        if self.mixed_precision and self.device == "cuda":
            with torch.cuda.amp.autocast():
                outputs, loss = self.loss_calculation(x)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
        else:
            outputs, loss = self.loss_calculation(x)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        return outputs



    
    

    def collect_params(self):
        """Collect the parameters from additional TTA layers and batch normalization layers.
        Updates both the `tta` layers and BatchNorm parameters while keeping others frozen.
        """
        params = []
        names = []

        
        for name, param in self.model.named_parameters():
            if ('buffer' in name) and param.requires_grad:
                params.append(param)
                names.append(name)

        #######################################################################
        #######################################################################

        if self.with_bn :
            for nm, m in self.model.named_modules():
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                    for np, p in m.named_parameters():
                        if np in ['weight', 'bias']:  # weight is scale, bias is shift
                            params.append(p)
                            names.append(f"{nm}.{np}")


        print(names, 'param_names')


        return params, names
    

    def configure_model(self):
        """Configure model for use with BufferTTA.
        Enables training mode and updates both `tta` layers and batch normalization layers.
        """
        self.model.train()
        self.model.requires_grad_(False)

        for name, param in self.model.named_parameters():
            if "buffer" in name:       # layer/module 이름에 'tta'가 들어가면 grad 활성화
                param.requires_grad_(True)

        # 2) BatchNorm 같은 모듈 처리는 여전히 module 기반으로 처리해야 함
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None

        if self.with_bn :
            for m in self.model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.requires_grad_(True)
                    # force use of batch stats in train and eval modes
                elif isinstance(m, nn.BatchNorm1d):
                    m.train()   # always forcing train mode in bn1d will cause problems for single sample tta
                    m.requires_grad_(True)
                elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
                    m.requires_grad_(True)

