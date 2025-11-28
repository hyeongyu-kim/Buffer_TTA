"""
Builds upon: https://github.com/mr-eggplant/EATA
Corresponding paper: https://arxiv.org/abs/2204.02610
"""

import os
import math
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from methods.base import TTAMethod
from datasets.data_loading import get_source_loader
from utils.registry import ADAPTATION_REGISTRY
from utils.losses import Entropy

import torchvision.transforms as transforms
logger = logging.getLogger(__name__)


@ADAPTATION_REGISTRY.register()
class EATA_Buffer(TTAMethod):
    """EATA adapts a model by entropy minimization during testing.
    Once EATAed, a model adapts itself by updating on every forward.
    """
    def __init__(self, cfg, model, num_classes):
        self.with_bn = cfg.WITH_BN
        super().__init__(cfg, model, num_classes)

        self.num_samples_update_1 = 0  # number of samples after first filtering, exclude unreliable samples
        self.num_samples_update_2 = 0  # number of samples after second filtering, exclude both unreliable and redundant samples
        self.e_margin = cfg.EATA.MARGIN_E0 * math.log(num_classes)   # hyper-parameter E_0 (Eqn. 3)
        self.d_margin = cfg.EATA.D_MARGIN   # hyperparameter \epsilon for cosine similarity thresholding (Eqn. 5)

        self.current_model_probs = None  # the moving average of probability vector (Eqn. 4)
        self.fisher_alpha = cfg.EATA.FISHER_ALPHA  # trade-off \beta for two losses (Eqn. 8)

        # setup loss function
        self.softmax_entropy = Entropy()

        if self.fisher_alpha > 0.0 and self.cfg.SOURCE.NUM_SAMPLES > 0:
            # compute fisher informatrix

            
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            transform = transforms.Compose([transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                normalize])






            batch_size_src = cfg.TEST.BATCH_SIZE if cfg.TEST.BATCH_SIZE > 1 else cfg.TEST.WINDOW_LENGTH
            _, fisher_loader = get_source_loader(dataset_name=cfg.CORRUPTION.DATASET,
                                                 adaptation=cfg.MODEL.ADAPTATION,
                                                #  preprocess=model.model_preprocess,
                                                 preprocess = transform,
                                                 data_root_dir=cfg.DATA_DIR,
                                                 batch_size=batch_size_src,
                                                 ckpt_path=cfg.MODEL.CKPT_PATH,
                                                 num_samples=cfg.SOURCE.NUM_SAMPLES,    # number of samples for ewc reg.
                                                 percentage=cfg.SOURCE.PERCENTAGE,
                                                 use_clip = cfg.MODEL.USE_CLIP,
                                                 workers=min(cfg.SOURCE.NUM_WORKERS, os.cpu_count()))
            ewc_optimizer = torch.optim.SGD(self.params, 0.001)
            self.fishers = {} # fisher regularizer items for anti-forgetting, need to be calculated pre model adaptation (Eqn. 9)
            train_loss_fn = nn.CrossEntropyLoss().to(self.device)
            for iter_, batch in enumerate(fisher_loader, start=1):
                images = batch[0].to(self.device, non_blocking=True)
                outputs = self.model(images)
                _, targets = outputs.max(1)
                loss = train_loss_fn(outputs, targets)
                loss.backward()
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        if iter_ > 1:
                            fisher = param.grad.data.clone().detach() ** 2 + self.fishers[name][0]
                        else:
                            fisher = param.grad.data.clone().detach() ** 2
                        if iter_ == len(fisher_loader):
                            fisher = fisher / iter_
                        self.fishers.update({name: [fisher, param.data.clone().detach()]})
                ewc_optimizer.zero_grad()
            logger.info("Finished computing the fisher matrices...")
            del ewc_optimizer
        else:
            logger.info("Not using EWC regularization. EATA decays to ETA!")
            self.fishers = None

    def loss_calculation(self, x):
        """Forward and adapt model on batch of data.
        Measure entropy of the model prediction, take gradients, and update params.
        """
        imgs_test = x[0]
        outputs = self.model(imgs_test)
        entropys = self.softmax_entropy(outputs)

        # filter unreliable samples
        filter_ids_1 = torch.where(entropys < self.e_margin)
        ids1 = filter_ids_1
        ids2 = torch.where(ids1[0] > -0.1)
        entropys = entropys[filter_ids_1]

        # filter redundant samples
        if self.current_model_probs is not None:
            cosine_similarities = F.cosine_similarity(self.current_model_probs.unsqueeze(dim=0), outputs[filter_ids_1].softmax(1), dim=1)
            filter_ids_2 = torch.where(torch.abs(cosine_similarities) < self.d_margin)
            entropys = entropys[filter_ids_2]
            updated_probs = update_model_probs(self.current_model_probs, outputs[filter_ids_1][filter_ids_2].softmax(1))
        else:
            updated_probs = update_model_probs(self.current_model_probs, outputs[filter_ids_1].softmax(1))
        coeff = 1 / (torch.exp(entropys.clone().detach() - self.e_margin))

        # implementation version 1, compute loss, all samples backward (some unselected are masked)
        entropys = entropys.mul(coeff)  # reweight entropy losses for diff. samples
        loss = entropys.mean(0)
        """
        # implementation version 2, compute loss, forward all batch, forward and backward selected samples again.
        # if x[ids1][ids2].size(0) != 0:
        #     loss = self.softmax_entropy(model(x[ids1][ids2])).mul(coeff).mean(0) # reweight entropy losses for diff. samples
        """
        if self.fishers is not None:
            ewc_loss = 0
            for name, param in self.model.named_parameters():
                if name in self.fishers:
                    ewc_loss += self.fisher_alpha * (self.fishers[name][0] * (param - self.fishers[name][1]) ** 2).sum()
            loss += ewc_loss

        self.num_samples_update_1 += filter_ids_1[0].size(0)
        self.num_samples_update_2 += entropys.size(0)
        self.current_model_probs = updated_probs
        perform_update = len(entropys) != 0
        return outputs, loss, perform_update

    @torch.enable_grad()
    def forward_and_adapt(self, x):
        if self.mixed_precision and self.device == "cuda":
            with torch.cuda.amp.autocast():
                outputs, loss, perform_update = self.loss_calculation(x)
            # update model only if not all instances have been filtered
            if perform_update:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            self.optimizer.zero_grad()
        else:
            outputs, loss, perform_update = self.loss_calculation(x)
            # update model only if not all instances have been filtered
            if perform_update:
                loss.backward()
                self.optimizer.step()
            self.optimizer.zero_grad()
        return outputs

    def reset(self):
        if self.model_states is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        self.load_model_and_optimizer()
        self.current_model_probs = None



    
    

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









def update_model_probs(current_model_probs, new_probs):
    if current_model_probs is None:
        if new_probs.size(0) == 0:
            return None
        else:
            with torch.no_grad():
                return new_probs.mean(0)
    else:
        if new_probs.size(0) == 0:
            with torch.no_grad():
                return current_model_probs
        else:
            with torch.no_grad():
                return 0.9 * current_model_probs + (1 - 0.9) * new_probs.mean(0)
