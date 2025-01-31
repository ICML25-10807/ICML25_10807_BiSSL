import torch
import torch.nn.functional as F
from torch import nn
import torch.distributed as dist
import timm
import numpy as np
import copy
import math


def Projector(arg_mlp, embedding, use_bn=True):
    mlp_spec = f"{embedding}-{arg_mlp}"
    layers = []
    f = list(map(int, mlp_spec.split("-")))
    for i in range(len(f) - 2):
        layers.append(nn.Linear(f[i], f[i + 1]))
        if use_bn:
            layers.append(nn.BatchNorm1d(f[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(f[-2], f[-1], bias=False))
    return nn.Sequential(*layers)


class MaxPool2dMPS(nn.MaxPool2d):
    """Pytorch only supports achieving second order gradients of maxpool2d on mps devices
    if return_indices=True, thus this minor modification.
    """

    def forward(self, input: torch.Tensor):
        return F.max_pool2d(
            input,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            ceil_mode=self.ceil_mode,
            return_indices=True,
        )[0]


def Backbone(arch, embedding=None):
    mod = timm.create_model(arch)

    if arch.startswith("resnet"):
        if embedding is None or embedding == mod.fc.in_features:
            embedding = mod.fc.in_features
            # mod.global_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
            mod.fc = nn.Flatten()  # nn.Identity()

            mod.maxpool = MaxPool2dMPS(
                kernel_size=mod.maxpool.kernel_size,
                stride=mod.maxpool.stride,
                padding=mod.maxpool.padding,
                dilation=mod.maxpool.dilation,
                ceil_mode=mod.maxpool.ceil_mode,
            )
        else:
            # If a custom embedding dimension is provided, we replace the final layer with a linear layer squeezing the output to the desired dimension
            mod.fc = nn.Linear(mod.fc.in_features, embedding)

            mod.maxpool = MaxPool2dMPS(
                kernel_size=mod.maxpool.kernel_size,
                stride=mod.maxpool.stride,
                padding=mod.maxpool.padding,
                dilation=mod.maxpool.dilation,
                ceil_mode=mod.maxpool.ceil_mode,
            )
    else:
        raise ValueError(f'Invalid architecture "{arch}".')

    return mod, embedding


class FullGatherLayer(torch.autograd.Function):
    """
    Gather tensors from all process and support backward propagation
    for the gradients across processes.
    """

    @staticmethod
    def forward(ctx, x):
        output = [
            torch.zeros_like(x) for _ in range(torch.distributed.get_world_size())
        ]
        torch.distributed.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        torch.distributed.all_reduce(all_gradients)
        return all_gradients[torch.distributed.get_rank()]


class NTXent(nn.Module):
    """
    Contrastive loss with distributed data parallel support.
    Based on https://github.com/AndrewAtanov/simclr-pytorch/blob/master/models/losses.py
    """

    LARGE_NUMBER = 1e9

    def __init__(self, tau=1.0, gpu=None, multiplier=2, distributed=False):
        super().__init__()
        self.tau = tau
        self.multiplier = multiplier
        self.distributed = distributed
        self.norm = 1.0

    def forward(self, z, get_map=False):
        n = z.shape[0]
        assert n % self.multiplier == 0

        z = F.normalize(z, p=2, dim=1) / np.sqrt(self.tau)

        if self.distributed:
            # This modified version assumes that the input z is a concatenated batch from all processes, obtained from
            # using all_gather prior to calling this function.

            # Split "back" into "all_gather format" list as [<proc0>, <proc1>, ...]
            z = torch.chunk(z, dist.get_world_size(), dim=0)

            # split it into [<proc0_aug0>, <proc0_aug1>, ..., <proc0_aug(m-1)>, <proc1_aug(m-1)>, ...]
            z_list = [chunk for x in z for chunk in x.chunk(self.multiplier)]

            # sort it to [<proc0_aug0>, <proc1_aug0>, ...] that simply means [<batch_aug0>, <batch_aug1>, ...] as expected below
            z_sorted = []
            for m in range(self.multiplier):
                for i in range(dist.get_world_size()):
                    z_sorted.append(z_list[i * self.multiplier + m])
            z = torch.cat(z_sorted, dim=0)
            n = z.shape[0]

        logits = z @ z.t()
        logits[np.arange(n), np.arange(n)] = -self.LARGE_NUMBER

        logprob = F.log_softmax(logits, dim=1)

        del logits

        # choose all positive objects for an example, for i it would be (i + k * n/m), where k=0...(m-1)
        m = self.multiplier
        labels = (np.repeat(np.arange(n), m) + np.tile(np.arange(m) * n // m, n)) % n
        # remove labels pointet to itself, i.e. (i, i)
        labels = labels.reshape(n, m)[:, 1:].reshape(-1)

        loss = (
            -logprob[np.repeat(np.arange(n), m - 1), labels].sum()
            / n
            / (m - 1)
            / self.norm
        )
        return loss, 0


#### SimCLR ####
class SimCLR(nn.Module):
    def __init__(
        self,
        arg_mlp,
        arg_arch,
        temp: float = 1,
        distributed=False,
        embedding_dim=None,
        return_embedding=False,
    ):
        super().__init__()

        self.backbone, self.num_features = Backbone(arg_arch, embedding=embedding_dim)
        self.projector = Projector(arg_mlp, self.num_features)

        self.criterion = nn.CrossEntropyLoss()

        self.repr_loss_fn = lambda x, y: 0
        self.repr_loss_fn = lambda x, y: 0
        self.cov_loss_fn = lambda x, y: 0

        self.distributed = distributed

        self.temp = temp

        self.nt_xent_loss = NTXent(tau=temp, distributed=distributed)
        self.return_embedding = return_embedding

    def forward(self, input_pair, alternative_backbone=None):

        x, y = input_pair
        if alternative_backbone is None:
            x_emb = self.backbone(x)
            y_emb = self.backbone(y)
        else:
            x_emb = alternative_backbone(x)
            y_emb = alternative_backbone(y)
        x = self.projector(x_emb)
        y = self.projector(y_emb)

        if self.distributed:
            z = torch.cat(FullGatherLayer.apply(torch.cat((x, y), dim=0)))
            loss, _ = self.nt_xent_loss(z)

        else:
            loss, _ = self.nt_xent_loss(torch.cat((x, y), dim=0))

        if self.return_embedding:
            return loss, (x_emb, y_emb)
        return loss


#### BYOL #####


def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val


class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new, beta=None):
        if beta is None:
            beta = self.beta
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class BYOL(nn.Module):
    def __init__(
        self,
        arg_mlp,
        arg_arch,
        distributed=True,
        embedding_dim=None,
        moving_average_decay=0.9995,
        ma_use_scheduler=False,
        ma_scheduler_length=None,
    ):
        super().__init__()

        self.backbone, self.num_features = Backbone(arg_arch, embedding=embedding_dim)

        self.projector = Projector(arg_mlp, self.num_features)
        self.predictor = Projector(arg_mlp, arg_mlp.split("-")[-1])

        self.backbone_target = copy.deepcopy(self.backbone)
        set_requires_grad(self.backbone_target, False)
        self.projector_target = copy.deepcopy(self.projector)
        set_requires_grad(self.projector_target, False)

        self.distributed = distributed

        self.target_ema_updater = EMA(moving_average_decay)
        self.ma_decay = moving_average_decay
        self.ma_use_scheduler = ma_use_scheduler
        self.scheduler_step = 1

        if ma_use_scheduler:
            assert (
                ma_scheduler_length is not None
            ), "Scheduler length must be provided if scheduler is used."
            # Should be equal to the total number of gradient updates, i.e. length of dataloader times number of epochs
            self.ma_scheduler_length = ma_scheduler_length

    def loss_fn(self, x, y):
        x = F.normalize(x, dim=-1, p=2)
        y = F.normalize(y, dim=-1, p=2)
        return 2 - 2 * (x * y).sum(dim=-1)

    def update_moving_average(self, beta=None):
        for current_module, ma_module in [
            (self.backbone, self.backbone_target),
            (self.projector, self.projector_target),
        ]:
            for current_params, ma_params in zip(
                current_module.parameters(), ma_module.parameters()
            ):
                old_weight, up_weight = ma_params.data, current_params.data
                ma_params.data = self.target_ema_updater.update_average(
                    old_weight, up_weight, beta=beta
                )

    def reinit_target_model(self):
        """
        Reinitialize the target model with the current model's weights.
        Necessary for BiSSL when we load a pre-trained model after the model has been initialised.
        """
        self.backbone_target = copy.deepcopy(self.backbone)
        set_requires_grad(self.backbone_target, False)
        self.projector_target = copy.deepcopy(self.projector)
        set_requires_grad(self.projector_target, False)

    def forward(self, input_pair, update_ema=True):

        if update_ema:
            self.update_moving_average(
                beta=(
                    None
                    if not self.ma_use_scheduler
                    else 1
                    - (1 - self.ma_decay)
                    * (
                        math.cos(
                            math.pi * self.scheduler_step / self.ma_scheduler_length
                        )
                        + 1
                    )
                    / 2
                )
            )
            if self.ma_use_scheduler:
                self.scheduler_step += 1
        set_requires_grad(self.backbone_target, False)
        set_requires_grad(self.projector_target, False)

        x, y = input_pair
        x_emb, y_emb = self.backbone(x), self.backbone(y)

        x_proj, y_proj = self.projector(x_emb), self.projector(y_emb)
        x_pred, y_pred = self.predictor(x_proj), self.predictor(y_proj)

        with torch.no_grad():
            x_emb_target, y_emb_target = self.backbone_target(x), self.backbone_target(
                y
            )
            x_proj_target, y_proj_target = (
                self.projector_target(x_emb_target).detach(),
                self.projector_target(y_emb_target).detach(),
            )

        if self.distributed:
            x_pred, y_pred = torch.cat(FullGatherLayer.apply(x_pred), dim=0), torch.cat(
                FullGatherLayer.apply(y_pred), dim=0
            )
            x_proj_target, y_proj_target = torch.cat(
                FullGatherLayer.apply(x_proj_target), dim=0
            ), torch.cat(FullGatherLayer.apply(y_proj_target), dim=0)

        loss_one = self.loss_fn(x_pred, y_proj_target.detach())
        loss_two = self.loss_fn(y_pred, x_proj_target.detach())
        loss = loss_one + loss_two

        return loss.mean()


class DSClassifier(nn.Module):
    def __init__(self, arg_arch, n_classes=10, embedding_dim=None):
        super().__init__()
        self.backbone, self.num_features = Backbone(arg_arch, embedding=embedding_dim)
        self.head = nn.Linear(self.num_features, n_classes)

    def forward(self, x):
        return self.head(self.backbone(x))
