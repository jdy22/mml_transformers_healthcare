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

import argparse
import os

import numpy as np
import torch
from networks.unetr_2d import UNETR_2D
from networks.unetr_2d_modality import UNETR_2D_modality
from networks.unetr_2d_organ import UNETR_2D_organ
from data_utils.data_loader import get_loader
from data_utils.data_loader_2 import get_loader_2
from data_utils.data_loader_3 import get_loader_3
from data_utils.visualise_data import plot_save_predictions

from monai.inferers import sliding_window_inference
from monai.utils.misc import set_determinism


parser = argparse.ArgumentParser(description="UNETR segmentation pipeline")
parser.add_argument(
    "--pretrained_dir", default="./runs_modality/run1/", type=str, help="pretrained checkpoint directory"
)
parser.add_argument("--data_dir", default="./amos22/", type=str, help="dataset directory")
parser.add_argument("--json_list", default="dataset_internal_val.json", type=str, help="dataset json file")
parser.add_argument(
    "--pretrained_model_name", default="model.pt", type=str, help="pretrained model name"
)
parser.add_argument(
    "--saved_checkpoint", default="ckpt", type=str, help="Supports torchscript or ckpt pretrained checkpoint type"
)
parser.add_argument("--mlp_dim", default=3072, type=int, help="mlp dimention in ViT encoder")
parser.add_argument("--hidden_size", default=768, type=int, help="hidden size dimention in ViT encoder")
parser.add_argument("--feature_size", default=64, type=int, help="feature size dimention")
parser.add_argument("--infer_overlap", default=0.5, type=float, help="sliding window inference overlap")
parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=16, type=int, help="number of output channels")
parser.add_argument("--num_heads", default=12, type=int, help="number of attention heads in ViT encoder")
parser.add_argument("--res_block", action="store_true", help="use residual blocks")
parser.add_argument("--conv_block", action="store_true", help="use conv blocks")
# parser.add_argument("--a_min", default=-175.0, type=float, help="a_min in ScaleIntensityRanged")
# parser.add_argument("--a_max", default=250.0, type=float, help="a_max in ScaleIntensityRanged")
parser.add_argument("--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged")
parser.add_argument("--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged")
parser.add_argument("--space_x", default=1.5, type=float, help="spacing in x direction")
parser.add_argument("--space_y", default=1.5, type=float, help="spacing in y direction")
parser.add_argument("--space_z", default=2.0, type=float, help="spacing in z direction")
parser.add_argument("--roi_x", default=112, type=int, help="roi size in x direction")
parser.add_argument("--roi_y", default=112, type=int, help="roi size in y direction")
parser.add_argument("--roi_z", default=1, type=int, help="roi size in z direction")
parser.add_argument("--dropout_rate", default=0.0, type=float, help="dropout rate")
parser.add_argument("--distributed", action="store_true", help="start distributed training")
parser.add_argument("--workers", default=8, type=int, help="number of workers")
# parser.add_argument("--RandFlipd_prob", default=0.2, type=float, help="RandFlipd aug probability")
# parser.add_argument("--RandRotate90d_prob", default=0.2, type=float, help="RandRotate90d aug probability")
# parser.add_argument("--RandScaleIntensityd_prob", default=0.1, type=float, help="RandScaleIntensityd aug probability")
# parser.add_argument("--RandShiftIntensityd_prob", default=0.1, type=float, help="RandShiftIntensityd aug probability")
parser.add_argument("--pos_embed", default="perceptron", type=str, help="type of position embedding")
parser.add_argument("--norm_name", default="instance", type=str, help="normalization layer type in decoder")
parser.add_argument("--lower", default=30.0, type=float, help="lower percentile in ScaleIntensityRangePercentilesd")
parser.add_argument("--upper", default=99.0, type=float, help="upper percentile in ScaleIntensityRangePercentilesd")
parser.add_argument("--train_samples", default=40, type=int, help="number of samples per training image")
parser.add_argument("--val_samples", default=20, type=int, help="number of samples per validation image")
parser.add_argument("--train_sampling", default="uniform", type=str, help="sampling distribution of organs during training")
parser.add_argument("--preprocessing", default=2, type=int, help="preprocessing option")
parser.add_argument("--data_augmentation", action="store_false", help="use data augmentation during training")
parser.add_argument("--additional_information", default="modality_concat", help="additional information provided to segmentation model")


def visualise_predictions(args, model, loader, modality, image_index, num_samples):
    for idx, batch in enumerate(loader):
        if idx == image_index:
            val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
            if args.additional_information == "modality_concat":
                val_outputs = sliding_window_inference(val_inputs, (args.roi_x, args.roi_y), 1, model, overlap=args.infer_overlap, modality=modality, info_mode="concat")
            if args.additional_information == "modality_concat2":
                val_outputs = sliding_window_inference(val_inputs, (args.roi_x, args.roi_y), 1, model, overlap=args.infer_overlap, modality=modality, info_mode="concat2")
            elif args.additional_information == "modality_add":
                val_outputs = sliding_window_inference(val_inputs, (args.roi_x, args.roi_y), 1, model, overlap=args.infer_overlap, modality=modality, info_mode="add")
            elif args.additional_information == "organ" or args.additional_information == "organ_inter" or args.additional_information == "organ_inter2" or args.additional_information == "organ_inter3" or args.additional_information == "organ_late":
                val_inputs_full = torch.cat((val_inputs, val_labels), dim=1)
                val_outputs = sliding_window_inference(val_inputs_full, (args.roi_x, args.roi_y), 1, model, overlap=args.infer_overlap)
            elif "organ_classif" in args.additional_information:
                val_outputs = sliding_window_inference(val_inputs, (args.roi_x, args.roi_y), 1, model, overlap=args.infer_overlap, test_mode=True, class_layer=args.classification_layer)
            else:
                val_outputs = sliding_window_inference(val_inputs, (args.roi_x, args.roi_y), 1, model, overlap=args.infer_overlap)
            val_outputs = torch.softmax(val_outputs, 1).cpu().numpy()
            val_outputs = np.argmax(val_outputs, axis=1, keepdims=True).astype(np.uint8)
            val_labels = val_labels.cpu().numpy()
            for sample in range(num_samples):
                x = val_inputs[sample, 0, :, :]
                y_pred = val_outputs[sample, 0, :, :]
                y_true = val_labels[sample, 0, :, :]
                plot_save_predictions(x, y_pred, y_true, image_index, sample, args, modality)
        

def main():
    args = parser.parse_args()
    args.test_mode = True
    args.test_type = "test" # "validation" or "test"
    if args.preprocessing == 1:
        val_loader = get_loader(args)
    elif args.preprocessing == 2:
        val_loader = get_loader_2(args)
    elif args.preprocessing == 3:
        val_loader = get_loader_3(args)
    loader_ct = val_loader[0]
    loader_mri = val_loader[1]
    pretrained_dir = args.pretrained_dir
    model_name = args.pretrained_model_name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_pth = os.path.join(pretrained_dir, model_name)
    if args.saved_checkpoint == "torchscript":
        model = torch.jit.load(pretrained_pth)
    elif args.saved_checkpoint == "ckpt":
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
        model_dict = torch.load(pretrained_pth)
        model.load_state_dict(model_dict["state_dict"])
    model.eval()
    model.to(device)

    with torch.no_grad():
        visualise_predictions(args, model, loader_ct, modality="CT", image_index=0, num_samples=5)
        visualise_predictions(args, model, loader_mri, modality="MRI", image_index=0, num_samples=5)


if __name__ == "__main__":
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

    set_determinism()

    main()