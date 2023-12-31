# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

### Main file which sets all of the arguments required for and runs the model training for the context-aware and baseline 2D UNETR models
# Adapted from the original UNETR code

import argparse
import os
from functools import partial

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.parallel
import torch.utils.data.distributed
from networks.unetr_2d import UNETR_2D
from networks.unetr_2d_modality import UNETR_2D_modality
from networks.unetr_2d_organ import UNETR_2D_organ
from networks.unetr_2d_clip import UNETR_2D_clip
from monai_research_contributions_main.UNETR.BTCV.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from trainer import run_training
from data_utils.data_loader import get_loader
from data_utils.data_loader_2 import get_loader_2
from data_utils.data_loader_3 import get_loader_3

from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss, DiceLoss
from monai.metrics import DiceMetric
from monai.transforms import Activations, AsDiscrete, Compose
from monai.utils.enums import MetricReduction

parser = argparse.ArgumentParser(description="UNETR segmentation pipeline")
parser.add_argument("--checkpoint", default=None, help="start training from saved checkpoint")
parser.add_argument("--logdir", default="run8-3b", type=str, help="directory to save the tensorboard logs")
parser.add_argument(
    "--pretrained_dir", default=None, type=str, help="pretrained checkpoint directory"
)
parser.add_argument("--data_dir", default="./amos22/", type=str, help="dataset directory")
parser.add_argument("--json_list", default="dataset_internal_val.json", type=str, help="dataset json file")
parser.add_argument(
    "--pretrained_model_name", default="model.pt", type=str, help="pretrained model name"
)
parser.add_argument("--save_checkpoint", action="store_false", help="save checkpoint during training")
parser.add_argument("--max_epochs", default=1000, type=int, help="max number of training epochs")
parser.add_argument("--batch_size", default=1, type=int, help="number of batch size")
parser.add_argument("--sw_batch_size", default=1, type=int, help="number of sliding window batch size")
parser.add_argument("--optim_lr", default=1e-4, type=float, help="optimization learning rate")
parser.add_argument("--optim_name", default="adamw", type=str, help="optimization algorithm")
parser.add_argument("--reg_weight", default=1e-5, type=float, help="regularization weight")
parser.add_argument("--momentum", default=0.99, type=float, help="momentum")
parser.add_argument("--noamp", action="store_true", help="do NOT use amp for training")
parser.add_argument("--val_every", default=20, type=int, help="validation frequency")
parser.add_argument("--distributed", action="store_true", help="start distributed training")
parser.add_argument("--world_size", default=1, type=int, help="number of nodes for distributed training")
parser.add_argument("--rank", default=0, type=int, help="node rank for distributed training")
parser.add_argument("--dist-url", default="tcp://127.0.0.1:23456", type=str, help="distributed url")
parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")
parser.add_argument("--workers", default=8, type=int, help="number of workers")
parser.add_argument("--model_name", default="unetr", type=str, help="model name")
parser.add_argument("--pos_embed", default="perceptron", type=str, help="type of position embedding")
parser.add_argument("--norm_name", default="instance", type=str, help="normalization layer type in decoder")
parser.add_argument("--num_heads", default=12, type=int, help="number of attention heads in ViT encoder")
parser.add_argument("--mlp_dim", default=3072, type=int, help="mlp dimention in ViT encoder")
parser.add_argument("--hidden_size", default=768, type=int, help="hidden size dimention in ViT encoder")
parser.add_argument("--feature_size", default=64, type=int, help="feature size dimention")
parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=16, type=int, help="number of output channels")
parser.add_argument("--res_block", action="store_true", help="use residual blocks")
parser.add_argument("--conv_block", action="store_true", help="use conv blocks")
parser.add_argument("--use_normal_dataset", action="store_true", help="use monai Dataset class")
# parser.add_argument("--a_min", default=-175.0, type=float, help="a_min in ScaleIntensityRanged")
# parser.add_argument("--a_max", default=250.0, type=float, help="a_max in ScaleIntensityRanged")
# parser.add_argument("--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged")
# parser.add_argument("--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged")
parser.add_argument("--space_x", default=1.5, type=float, help="spacing in x direction")
parser.add_argument("--space_y", default=1.5, type=float, help="spacing in y direction")
parser.add_argument("--space_z", default=2.0, type=float, help="spacing in z direction")
parser.add_argument("--roi_x", default=112, type=int, help="roi size in x direction")
parser.add_argument("--roi_y", default=112, type=int, help="roi size in y direction")
parser.add_argument("--roi_z", default=1, type=int, help="roi size in z direction")
parser.add_argument("--dropout_rate", default=0.0, type=float, help="dropout rate")
# parser.add_argument("--RandFlipd_prob", default=0.2, type=float, help="RandFlipd aug probability")
# parser.add_argument("--RandRotate90d_prob", default=0.2, type=float, help="RandRotate90d aug probability")
# parser.add_argument("--RandScaleIntensityd_prob", default=0.1, type=float, help="RandScaleIntensityd aug probability")
# parser.add_argument("--RandShiftIntensityd_prob", default=0.1, type=float, help="RandShiftIntensityd aug probability")
parser.add_argument("--infer_overlap", default=0.5, type=float, help="sliding window inference overlap")
parser.add_argument("--lrschedule", default="warmup_cosine", type=str, help="type of learning rate scheduler")
parser.add_argument("--warmup_epochs", default=50, type=int, help="number of warmup epochs")
parser.add_argument("--resume_ckpt", action="store_true", help="resume training from pretrained checkpoint")
parser.add_argument("--resume_jit", action="store_true", help="resume training from pretrained torchscript checkpoint")
parser.add_argument("--smooth_dr", default=1e-6, type=float, help="constant added to dice denominator to avoid nan")
parser.add_argument("--smooth_nr", default=0.0, type=float, help="constant added to dice numerator to avoid zero")
parser.add_argument("--lower", default=30.0, type=float, help="lower percentile in ScaleIntensityRangePercentilesd for preprocessing option 1")
parser.add_argument("--upper", default=99.0, type=float, help="upper percentile in ScaleIntensityRangePercentilesd for preprocessing option 1")
parser.add_argument("--train_samples", default=40, type=int, help="number of samples per training image")
parser.add_argument("--val_samples", default=20, type=int, help="number of samples per validation image")
parser.add_argument("--train_sampling", default="uniform", type=str, help="sampling distribution of organs during training")
parser.add_argument("--preprocessing", default=2, type=int, help="preprocessing option")
parser.add_argument("--data_augmentation", action="store_false", help="use data augmentation during training")
parser.add_argument("--additional_information", default="organ_inter2", help="additional information provided to segmentation model")
parser.add_argument("--loss_combination_factor", default=1.0, type=float, help="combination factor for segmentation and classification losses")
parser.add_argument("--classification_layer", default=12, type=int, help="Transformer layer for classification")


def main():
    args = parser.parse_args()
    args.amp = not args.noamp
    if "modality" in args.additional_information:
        args.logdir = "./runs_modality/" + args.logdir
    elif "organ" in args.additional_information:
        args.logdir = "./runs_organ/" + args.logdir
    elif "clip" in args.additional_information:
        args.logdir = "./runs_clip/" + args.logdir
    else:
        args.logdir = "./runs/" + args.logdir

    if args.distributed:
        args.ngpus_per_node = torch.cuda.device_count()
        print("Found total gpus", args.ngpus_per_node)
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args,))
    else:
        main_worker(gpu=0, args=args)


def main_worker(gpu, args):
    if args.distributed:
        torch.multiprocessing.set_start_method("fork", force=True)
    np.set_printoptions(formatter={"float": "{: 0.3f}".format}, suppress=True)
    args.gpu = gpu
    if args.distributed:
        args.rank = args.rank * args.ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank
        )
    torch.cuda.set_device(args.gpu)
    torch.backends.cudnn.benchmark = True
    args.test_mode = False
    if args.preprocessing == 1:
        loader = get_loader(args)
    elif args.preprocessing == 2:
        loader = get_loader_2(args)
    elif args.preprocessing == 3:
        loader = get_loader_3(args)
    print(args.rank, " gpu", args.gpu)
    if args.rank == 0:
        print("Batch size is:", args.batch_size, "epochs", args.max_epochs)
    inf_size = [args.roi_x, args.roi_y]
    pretrained_dir = args.pretrained_dir
    if (args.model_name is None) or args.model_name == "unetr":
        if args.additional_information == "modality_concat" or args.additional_information == "modality_concat2" or args.additional_information == "modality_add":
            model = UNETR_2D_modality(
                in_channels=args.in_channels,
                out_channels=args.out_channels,
                img_size=(args.roi_x, args.roi_y),
                feature_size=args.feature_size,
                hidden_size=args.hidden_size,
                mlp_dim=args.mlp_dim,
                num_heads=args.num_heads,
                pos_embed=args.pos_embed,
                norm_name=args.norm_name,
                conv_block=True,
                res_block=True,
                dropout_rate=args.dropout_rate,
                separate_decoders=False,
            )
        elif "modality_decoder" in args.additional_information:
            model = UNETR_2D_modality(
                in_channels=args.in_channels,
                out_channels=args.out_channels,
                img_size=(args.roi_x, args.roi_y),
                feature_size=args.feature_size,
                hidden_size=args.hidden_size,
                mlp_dim=args.mlp_dim,
                num_heads=args.num_heads,
                pos_embed=args.pos_embed,
                norm_name=args.norm_name,
                conv_block=True,
                res_block=True,
                dropout_rate=args.dropout_rate,
                separate_decoders=True,
            )
        elif args.additional_information == "organ":
            model = UNETR_2D_organ(
                in_channels=args.in_channels,
                out_channels=args.out_channels,
                img_size=(args.roi_x, args.roi_y),
                feature_size=args.feature_size,
                hidden_size=args.hidden_size,
                mlp_dim=args.mlp_dim,
                num_heads=args.num_heads,
                pos_embed=args.pos_embed,
                norm_name=args.norm_name,
                conv_block=True,
                res_block=True,
                dropout_rate=args.dropout_rate,
                info_mode="early",
            )
        elif args.additional_information == "organ_classif":
            model = UNETR_2D_organ(
                in_channels=args.in_channels,
                out_channels=args.out_channels,
                img_size=(args.roi_x, args.roi_y),
                feature_size=args.feature_size,
                hidden_size=args.hidden_size,
                mlp_dim=args.mlp_dim,
                num_heads=args.num_heads,
                pos_embed=args.pos_embed,
                norm_name=args.norm_name,
                conv_block=True,
                res_block=True,
                dropout_rate=args.dropout_rate,
                info_mode="classif",
            )
        elif args.additional_information == "organ_classif_early":
            model = UNETR_2D_organ(
                in_channels=args.in_channels,
                out_channels=args.out_channels,
                img_size=(args.roi_x, args.roi_y),
                feature_size=args.feature_size,
                hidden_size=args.hidden_size,
                mlp_dim=args.mlp_dim,
                num_heads=args.num_heads,
                pos_embed=args.pos_embed,
                norm_name=args.norm_name,
                conv_block=True,
                res_block=True,
                dropout_rate=args.dropout_rate,
                info_mode="classif_early",
            )
        elif args.additional_information == "organ_classif_inter3":
            model = UNETR_2D_organ(
                in_channels=args.in_channels,
                out_channels=args.out_channels,
                img_size=(args.roi_x, args.roi_y),
                feature_size=args.feature_size,
                hidden_size=args.hidden_size,
                mlp_dim=args.mlp_dim,
                num_heads=args.num_heads,
                pos_embed=args.pos_embed,
                norm_name=args.norm_name,
                conv_block=True,
                res_block=True,
                dropout_rate=args.dropout_rate,
                info_mode="classif_inter3",
            )
        elif args.additional_information == "organ_inter":
            model = UNETR_2D_organ(
                in_channels=args.in_channels,
                out_channels=args.out_channels,
                img_size=(args.roi_x, args.roi_y),
                feature_size=args.feature_size,
                hidden_size=args.hidden_size,
                mlp_dim=args.mlp_dim,
                num_heads=args.num_heads,
                pos_embed=args.pos_embed,
                norm_name=args.norm_name,
                conv_block=True,
                res_block=True,
                dropout_rate=args.dropout_rate,
                info_mode="inter",
            )
        elif args.additional_information == "organ_inter2":
            model = UNETR_2D_organ(
                in_channels=args.in_channels,
                out_channels=args.out_channels,
                img_size=(args.roi_x, args.roi_y),
                feature_size=args.feature_size,
                hidden_size=args.hidden_size,
                mlp_dim=args.mlp_dim,
                num_heads=args.num_heads,
                pos_embed=args.pos_embed,
                norm_name=args.norm_name,
                conv_block=True,
                res_block=True,
                dropout_rate=args.dropout_rate,
                info_mode="inter2",
            )
        elif args.additional_information == "organ_inter3":
            model = UNETR_2D_organ(
                in_channels=args.in_channels,
                out_channels=args.out_channels,
                img_size=(args.roi_x, args.roi_y),
                feature_size=args.feature_size,
                hidden_size=args.hidden_size,
                mlp_dim=args.mlp_dim,
                num_heads=args.num_heads,
                pos_embed=args.pos_embed,
                norm_name=args.norm_name,
                conv_block=True,
                res_block=True,
                dropout_rate=args.dropout_rate,
                info_mode="inter3",
            )
        elif args.additional_information == "organ_late":
            model = UNETR_2D_organ(
                in_channels=args.in_channels,
                out_channels=args.out_channels,
                img_size=(args.roi_x, args.roi_y),
                feature_size=args.feature_size,
                hidden_size=args.hidden_size,
                mlp_dim=args.mlp_dim,
                num_heads=args.num_heads,
                pos_embed=args.pos_embed,
                norm_name=args.norm_name,
                conv_block=True,
                res_block=True,
                dropout_rate=args.dropout_rate,
                info_mode="late",
            )
        elif args.additional_information == "clip_early":
            model = UNETR_2D_clip(
                in_channels=args.in_channels,
                out_channels=args.out_channels,
                img_size=(args.roi_x, args.roi_y),
                feature_size=args.feature_size,
                hidden_size=args.hidden_size,
                mlp_dim=args.mlp_dim,
                num_heads=args.num_heads,
                pos_embed=args.pos_embed,
                norm_name=args.norm_name,
                conv_block=True,
                res_block=True,
                dropout_rate=args.dropout_rate,
                info_mode="early",
            )
        elif args.additional_information == "clip_late":
            model = UNETR_2D_clip(
                in_channels=args.in_channels,
                out_channels=args.out_channels,
                img_size=(args.roi_x, args.roi_y),
                feature_size=args.feature_size,
                hidden_size=args.hidden_size,
                mlp_dim=args.mlp_dim,
                num_heads=args.num_heads,
                pos_embed=args.pos_embed,
                norm_name=args.norm_name,
                conv_block=True,
                res_block=True,
                dropout_rate=args.dropout_rate,
                info_mode="late",
            )
        else:
            model = UNETR_2D(
                in_channels=args.in_channels,
                out_channels=args.out_channels,
                img_size=(args.roi_x, args.roi_y),
                feature_size=args.feature_size,
                hidden_size=args.hidden_size,
                mlp_dim=args.mlp_dim,
                num_heads=args.num_heads,
                pos_embed=args.pos_embed,
                norm_name=args.norm_name,
                conv_block=True,
                res_block=True,
                dropout_rate=args.dropout_rate,
            )

        if args.resume_ckpt:
            model_dict = torch.load(os.path.join(pretrained_dir, args.pretrained_model_name))
            model.load_state_dict(model_dict["state_dict"])
            print("Use pretrained weights")

        if args.resume_jit:
            if not args.noamp:
                print("Training from pre-trained checkpoint does not support AMP\nAMP is disabled.")
                args.amp = args.noamp
            model = torch.jit.load(os.path.join(pretrained_dir, args.pretrained_model_name))
    else:
        raise ValueError("Unsupported model " + str(args.model_name))

    dice_loss = DiceCELoss(to_onehot_y=True, softmax=True, squared_pred=True, smooth_nr=args.smooth_nr, smooth_dr=args.smooth_dr)
    if "organ_classif" in args.additional_information:
        bce_loss = torch.nn.BCEWithLogitsLoss()
        loss_func = [dice_loss, bce_loss]
    else:
        loss_func = dice_loss
    post_label = AsDiscrete(to_onehot=args.out_channels)
    post_pred = AsDiscrete(argmax=True, to_onehot=args.out_channels)
    dice_acc = DiceMetric(include_background=True, reduction=MetricReduction.MEAN, get_not_nans=True)
    model_inferer = partial(
        sliding_window_inference,
        roi_size=inf_size,
        sw_batch_size=args.sw_batch_size,
        predictor=model,
        overlap=args.infer_overlap,
    )

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total parameters count", pytorch_total_params)

    best_acc = 0
    start_epoch = 0

    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        from collections import OrderedDict

        new_state_dict = OrderedDict()
        for k, v in checkpoint["state_dict"].items():
            new_state_dict[k.replace("backbone.", "")] = v
        model.load_state_dict(new_state_dict, strict=False)
        if args.additional_information == "modality_decoder_pretrained":
            model.encoder1b.load_state_dict(model.encoder1.state_dict())
            model.encoder2b.load_state_dict(model.encoder2.state_dict())
            model.encoder3b.load_state_dict(model.encoder3.state_dict())
            model.encoder4b.load_state_dict(model.encoder4.state_dict())
            model.decoder5b.load_state_dict(model.decoder5.state_dict())
            model.decoder4b.load_state_dict(model.decoder4.state_dict())
            model.decoder3b.load_state_dict(model.decoder3.state_dict())
            model.decoder2b.load_state_dict(model.decoder2.state_dict())
            model.outb.load_state_dict(model.out.state_dict())
        if "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"] + 1
        if "best_acc" in checkpoint:
            best_acc = checkpoint["best_acc"]
        print("=> loaded checkpoint '{}' (epoch {}) (bestacc {})".format(args.checkpoint, start_epoch, best_acc))

    model.cuda(args.gpu)

    if args.distributed:
        torch.cuda.set_device(args.gpu)
        if args.norm_name == "batch":
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model.cuda(args.gpu)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], output_device=args.gpu, find_unused_parameters=True
        )
    if args.optim_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight)
    elif args.optim_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight)
    elif args.optim_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.optim_lr, momentum=args.momentum, nesterov=True, weight_decay=args.reg_weight
        )
    else:
        raise ValueError("Unsupported Optimization Procedure: " + str(args.optim_name))

    if args.lrschedule == "warmup_cosine":
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer, warmup_epochs=args.warmup_epochs, max_epochs=args.max_epochs
        )
    elif args.lrschedule == "cosine_anneal":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)
        if args.checkpoint is not None:
            scheduler.step(epoch=start_epoch)
    else:
        scheduler = None
    accuracy = run_training(
        model=model,
        train_loader=loader[0],
        val_loader=loader[1],
        optimizer=optimizer,
        loss_func=loss_func,
        acc_func=dice_acc,
        args=args,
        model_inferer=model_inferer,
        scheduler=scheduler,
        start_epoch=start_epoch,
        post_label=post_label,
        post_pred=post_pred,
    )
    return accuracy


if __name__ == "__main__":
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

    main()

    # from monai.utils.misc import set_determinism
    # set_determinism()
    
    # args = parser.parse_args()
    # args.test_mode = True
    # args.test_type = "validation"
    # args.data_dir = "/Users/joannaye/Documents/_Imperial_AI_MSc/1_Individual_project/AMOS_dataset/amos22/"
    # args.json_list = "dataset_small.json"
    # loader = get_loader_2(args)

    # train_loader=loader[0]
    # for idx, batch_data in enumerate(train_loader):
    #     if isinstance(batch_data, list):
    #         data, target = batch_data
    #     else:
    #         data, target = batch_data["image"], batch_data["label"]
    #     print(idx)
    #     print(data.shape)
    #     print(target.shape)

    # # Visualisation
    # from data_utils.visualise_data import display_2d_tensor, plot_intensity_histogram_from_tensor_ct_mri

    # val_loader_ct = loader[0]
    # for idx, batch_data in enumerate(val_loader_ct):
    #     if isinstance(batch_data, list):
    #         data_ct, target_ct = batch_data
    #     else:
    #         data_ct, target_ct = batch_data["image"], batch_data["label"]
    #     print(idx)
    #     # print(data_ct.shape)
    #     # print(target_ct.shape)
    #     print(torch.unique(target_ct[0]))

    # val_loader_mri = loader[1]
    # for idx, batch_data in enumerate(val_loader_mri):
    #     if isinstance(batch_data, list):
    #         data_mri, target_mri = batch_data
    #     else:
    #         data_mri, target_mri = batch_data["image"], batch_data["label"]
    #     print(idx)
    #     # print(data_mri.shape)
    #     # print(target_mri.shape)
    #     print(torch.unique(target_mri[0]))

    # # Visualisation
    # image_ct = data_ct[0, 0, :, :]
    # image_mri = data_mri[0, 0, :, :]
    # plot_intensity_histogram_from_tensor_ct_mri(image_ct, image_mri)