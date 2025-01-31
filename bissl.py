import os
import sys
import json

import argparse

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import InterpolationMode

from utils.models import SimCLR, BYOL, DSClassifier

import utils.augmentations as aug
from utils.distributed import init_distributed_mode

from utils.bissl_trainer import BiSSL_Trainer, IGGradCalc, cg_solver

from utils.parsers import get_args_bissl
from utils.training_fn import (
    print_and_save_stat,
    test_d,
    test_d_mAP,
    CosineLRSchedulerWithWarmup,
    train_d_classic,
)

from utils.datasets import GetData
from utils.optimizers import get_optimizer


if __name__ == "__main__":
    parser = get_args_bissl()
    args = parser.parse_args()

    ### Distributed Setup ###
    torch.backends.cudnn.benchmark = True

    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["OMP_NUM_THREADS"] = args.omp_num_threads
    args.world_size = torch.cuda.device_count()
    init_distributed_mode(args)
    device = torch.device(args.device)

    print_and_save_stat("(BiSSL) " + " ".join(sys.argv), rank=args.rank)

    print_and_save_stat(
        json.dumps(dict(train_type="Console Args", data=" ".join(sys.argv))),
        print_in_console=False,
        rank=args.rank,
    )

    print_and_save_stat("", rank=args.rank)
    print_and_save_stat(f"Device = {device}", rank=args.rank)

    #### DATA PREP ####
    get_data = GetData(args.data_dir, args.download_dataset)
    interpolation = InterpolationMode.BICUBIC

    data_p = get_data(
        dset_name=args.p_dset,
        split="train" if args.p_dset != "stl10" else "unlabeled",
        transform=aug.TrainTransform(
            img_size=args.img_size,
            interpolation_mode=interpolation,
            min_ratio=args.img_crop_min_ratio,
        ),
    )

    data_d_train = get_data(
        dset_name=args.d_dset,
        split="train",
        transform=aug.BLODownstreamTrainTransform(
            img_size=args.img_size,
            interpolation_mode=interpolation,
            min_ratio=args.img_crop_min_ratio,
        ),
    )

    data_d_test = get_data(
        dset_name=args.d_dset,
        split="val",
        transform=aug.BLODownstreamTestTransform(
            img_size=args.img_size, interpolation_mode=interpolation
        ),
    )

    assert data_d_train.classes == data_d_test.classes

    print_and_save_stat("Using downstream dataset: " + args.d_dset, rank=args.rank)

    sampler_p = torch.utils.data.distributed.DistributedSampler(
        data_p, shuffle=True, drop_last=True
    )
    sampler_d_train = torch.utils.data.distributed.DistributedSampler(
        data_d_train,
        shuffle=True,
        drop_last=True,
    )
    sampler_d_test = torch.utils.data.distributed.DistributedSampler(
        data_d_test,
        shuffle=True,
        drop_last=False,
    )

    assert args.p_batch_size % args.world_size == 0
    assert args.d_batch_size % args.world_size == 0
    per_device_p_batch_size = args.p_batch_size // args.world_size
    per_device_d_batch_size = args.d_batch_size // args.world_size

    # Create data loaders.
    # For training
    dataloader_p = DataLoader(
        data_p,
        batch_size=per_device_p_batch_size,
        num_workers=args.num_workers,
        sampler=sampler_p,
        pin_memory=True,
    )

    dataloader_d_train = DataLoader(
        data_d_train,
        batch_size=per_device_d_batch_size,
        num_workers=args.num_workers,
        sampler=sampler_d_train,
        pin_memory=True,
    )

    # If number of steps specified is larger than the length of the dataloader,
    # the current implementation changes the iter type to epochs (i.e. upper_num_iter
    # specifies number of upper level epochs to conduct instead of gradient steps),
    # and adjusts the number of epochs accordingly (see todo below).
    # TODO: Currently this rounds the number of steps down to the nearest multiple
    # of the dataloader length. This is no issue for the datasets used in the
    # experiments, but may cause unwanted behaviour for other very small scale datasets.

    if args.upper_num_iter >= len(dataloader_d_train):
        args.upper_num_iter = args.upper_num_iter // len(dataloader_d_train)
        args.upper_iter_type = "epochs"
    else:
        args.upper_iter_type = "steps"

    dataloader_d_test = DataLoader(
        data_d_test,
        batch_size=args.d_batch_size,
        num_workers=args.num_workers,
        sampler=sampler_d_test,
        pin_memory=True,
    )

    ### MODEL IMPORT ###
    match args.p_pretext_type:
        case "simclr":
            model_p = SimCLR(
                args.p_mlp,
                args.arch,
                temp=args.p_temperature,
                distributed=True,
            )
        case "byol":
            model_p = BYOL(
                args.p_mlp,
                args.arch,
                distributed=True,
                moving_average_decay=args.p_ma_decay,
                ma_use_scheduler=args.p_ma_use_scheduler,
                ma_scheduler_length=len(dataloader_p) * args.epochs,
            )

    print_and_save_stat(
        f"Backbone: {sum(p.numel() for p in model_p.backbone.parameters() if p.requires_grad)/1_000_000} M Parameters",
        rank=args.rank,
    )
    print_and_save_stat(
        f"Pretext Head: {sum(p.numel() for p in model_p.projector.parameters() if p.requires_grad)/1_000_000} M Parameters",
        rank=args.rank,
    )
    if args.p_pretext_type == "byol":
        print_and_save_stat(
            f"Pretext Head (BYOL Predictor): {sum(p.numel() for p in model_p.predictor.parameters() if p.requires_grad)/1_000_000} M Parameters",
            rank=args.rank,
        )

    n_classes = len(data_d_test.classes)

    # Defines linear warmup model if a linear warmup is requested and a path to a pre-trained backbone is specified.
    # Otherwise we initialize with the upper-level model.
    if args.d_linear_warmup_epochs > 0:
        model_d_warmup = DSClassifier(
            args.arch,
            n_classes=n_classes,
        )
        print_and_save_stat(
            f"Downstream Head: {sum(p.numel() for p in model_d_warmup.head.parameters() if p.requires_grad)/1_000_000} M Parameters",
            rank=args.rank,
        )
    else:
        model_d = DSClassifier(
            args.arch,
            n_classes=n_classes,
        )
        print_and_save_stat(
            f"Downstream Head: {sum(p.numel() for p in model_d.head.parameters() if p.requires_grad)/1_000_000} M Parameters",
            rank=args.rank,
        )

    ### (Partial) TRAINING SETUP  ###
    loss_fn_d = torch.nn.CrossEntropyLoss().to(device)

    optimizer_p = get_optimizer(
        args.p_optimizer,
        model_p.parameters(),
        lr=args.p_lr,
        wd=args.p_wd,
        momentum=args.p_momentum,
    )

    lr_scheduler_p = CosineLRSchedulerWithWarmup(
        optimizer=optimizer_p,
        n_epochs=0,
        len_loader=0,
        warmup_epochs=0,
    )
    # As the warmup and remainder of training have different number of steps,
    # we set the total_steps and warmup_steps of the lr scheduler manually as follows.
    lr_scheduler_p.total_steps = (args.lower_num_iter * args.epochs) * len(dataloader_p)
    lr_scheduler_p.warmup_steps = 10 * len(dataloader_p)

    # The solver for approximating the upper-level gradient.
    # Using CG, as described in the paper.
    blo_grad_calc = IGGradCalc(
        solver=cg_solver,
        lam_dampening=args.cg_lam_dampening,
        solver_kwargs=dict(
            iter_num=args.cg_iter_num,
            verbose=bool(args.cg_verbose),
        ),
        lower_update_ema=False if args.p_pretext_type == "byol" else None,
    )

    backbone_tostore_path = (
        args.model_dir
        + "BiSSL-backbone_"
        + args.p_pretext_type
        + "_arch-"
        + args.arch
        + f"_dset-{args.d_dset}.pth"
    )

    best_acc_dbb = argparse.Namespace(top1=0, top5=0)
    best_acc_pbb = argparse.Namespace(top1=0, top5=0)

    model_p.to(device)

    # Loads the SSL pre-trained lower-level model.
    if args.rank == 0:
        for i in range(args.world_size):
            model_p.backbone.load_state_dict(
                torch.load(
                    args.model_dir + args.p_pretrained_backbone_filename,
                    map_location=f"cuda:{i}",
                )
            )
            model_p.projector.load_state_dict(
                torch.load(
                    args.model_dir + args.p_pretrained_proj_filename,
                    map_location=f"cuda:{i}",
                )
            )
            if args.p_pretext_type == "byol":
                model_p.predictor.load_state_dict(
                    torch.load(
                        args.model_dir + args.p_pretrained_pred_filename,
                        map_location=f"cuda:{i}",
                    )
                )
    if args.p_pretext_type == "byol":
        # Re-initialises target model s.t. its parameters are equal to the online model.
        model_p.reinit_target_model()

    # ###############################
    # ######## LINEAR WARMUP  #######
    # ###############################
    if args.d_linear_warmup_epochs > 0:
        model_d_warmup.backbone.load_state_dict(model_p.backbone.state_dict())

        w_optim = get_optimizer(
            args.d_optimizer,
            params=model_d_warmup.head.parameters(),
            lr=args.d_lr,
            wd=args.d_wd,
            momentum=args.d_momentum,
        )

        best_acc_lw = argparse.Namespace(top1=0, top5=0)

        # Freeze the backbone during linear head warmup.
        for par in model_d_warmup.backbone.parameters():
            par.requires_grad = False

        model_d_warmup.to(device)
        model_d_warmup = torch.nn.parallel.DistributedDataParallel(model_d_warmup)

        print_and_save_stat("Linear Warmup Training", rank=args.rank)
        for w_epoch in range(args.d_linear_warmup_epochs):
            print_and_save_stat("", rank=args.rank)
            print_and_save_stat(
                f"Warmup Epoch {w_epoch+1} / {args.d_linear_warmup_epochs}\n-------------------------------",
                rank=args.rank,
            )

            train_d_classic(
                model=model_d_warmup,
                loss_fn=loss_fn_d,
                dataloader=dataloader_d_train,
                optimizer=w_optim,
                device=device,
                rank=args.rank,
            )

            if (w_epoch + 1) % (args.d_linear_warmup_epochs // 5) == 0:
                print_and_save_stat("", rank=args.rank)
                if args.d_dset == "voc2007":
                    top1_lw_tr = test_d_mAP(
                        dataloader=dataloader_d_test,
                        model=model_d_warmup,
                        device=device,
                        label="Test Acc (LW, mAP)",
                        rank=args.rank,
                    )
                    top5_lw_tr, loss_lw_tr = 0, 0
                else:

                    top1_lw_tr, top5_lw_tr, loss_lw_tr = test_d(
                        dataloader=dataloader_d_test,
                        model=model_d_warmup,
                        loss_fn=loss_fn_d,
                        device=device,
                        label="Test Acc (LW)",
                        rank=args.rank,
                    )

                best_acc_lw.top1 = max(best_acc_lw.top1, top1_lw_tr)
                best_acc_lw.top5 = max(best_acc_lw.top5, top5_lw_tr)

        print_and_save_stat("", rank=args.rank)
        print_and_save_stat("FINISHED Linear Warmup Training", rank=args.rank)
        print_and_save_stat("", rank=args.rank)

        model_d = DSClassifier(args.arch, n_classes=n_classes)

        model_d.load_state_dict(model_d_warmup.module.state_dict())

        for par_d in model_d.parameters():
            par_d.requires_grad = True

        del model_d_warmup
        del w_optim
        del best_acc_lw

    #### BiSSL TRAINING SETUP ####
    optimizer_d = get_optimizer(
        args.d_optimizer,
        model_d.parameters(),
        lr=args.d_lr,
        wd=args.d_wd,
        momentum=args.d_momentum,
    )

    lr_scheduler_d = CosineLRSchedulerWithWarmup(
        optimizer=optimizer_d,
        n_epochs=(
            args.epochs
            if args.upper_iter_type == "steps"
            else args.epochs * args.upper_num_iter
        ),
        len_loader=(
            args.upper_num_iter
            if args.upper_iter_type == "steps"
            else len(dataloader_d_train)
        ),
        warmup_epochs=0,
    )

    model_d.to(device)

    # In case of no linear warmup, the backbone is not yet loaded with the pre-trained model.
    if args.d_linear_warmup_epochs == 0:
        model_d.backbone.load_state_dict(model_p.module.backbone.state_dict())
    model_d = torch.nn.parallel.DistributedDataParallel(model_d)
    model_p = torch.nn.parallel.DistributedDataParallel(model_p)

    bissl_trainer = BiSSL_Trainer(
        optimizers=(optimizer_d, optimizer_p),
        models=(model_d, model_p),
        loss_fn_d=loss_fn_d,
        device=device,
        blo_grad_calc=blo_grad_calc,
        rank=args.rank,
        lr_scheduler_p=lr_scheduler_p,
        lr_scheduler_d=lr_scheduler_d,
        lower_num_iter=args.lower_num_iter,
        upper_num_iter=args.upper_num_iter,
        upper_iter_type=args.upper_iter_type,
    )

    print_and_save_stat("", rank=args.rank)
    print_and_save_stat("BiSSL Training", rank=args.rank)

    # ###############################
    # ########### BiSSL  ############
    # ###############################

    for epoch in range(args.epochs):
        print_and_save_stat("", rank=args.rank)
        print_and_save_stat(
            f"Epoch {epoch+1} / {args.epochs}\n-------------------------------",
            rank=args.rank,
        )

        log_dicts = bissl_trainer(
            dataloaders=(dataloader_d_train, dataloader_p),
            samplers=(sampler_d_train, sampler_p),
            epoch=epoch,
            lambd=args.lam,
        )

        print_and_save_stat("", rank=args.rank)

        if args.d_dset == "voc2007":
            top1_dbb = test_d_mAP(
                dataloader_d_test,
                model_d,
                device,
                label="Test Error (D Backbone, mAP)",
                rank=args.rank,
            )
            top5_dbb, loss_dbb = 0, 0
        else:
            top1_dbb, top5_dbb, loss_dbb = test_d(
                dataloader_d_test,
                model_d,
                loss_fn_d,
                device,
                label="Test Error (D Backbone)",
                rank=args.rank,
            )

        best_acc_top1_prev = best_acc_dbb.top1

        best_acc_dbb.top1 = max(best_acc_dbb.top1, top1_dbb)
        best_acc_dbb.top5 = max(best_acc_dbb.top5, top5_dbb)

        if args.rank == 0 and args.save_model:
            model_p.eval()
            sd_p_bb = model_p.module.backbone.state_dict()
            torch.save(sd_p_bb, backbone_tostore_path)

            print_and_save_stat("Lower-level/Pretext Backbone Saved.", rank=args.rank)
            print_and_save_stat("", rank=args.rank)
