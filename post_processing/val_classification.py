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

### Code to evaluate the classification accuracy of a joint segmentation and classification model
# Adapted from the original UNETR code

import argparse
import os

import numpy as np
import torch
from networks.unetr_2d_organ import UNETR_2D_organ
# from data_utils.data_loader import get_loader
from data_utils.data_loader_2 import get_loader_2
# from data_utils.data_loader_3 import get_loader_3

from monai.inferers import sliding_window_inference
from monai.utils.misc import set_determinism

parser = argparse.ArgumentParser(description="UNETR segmentation pipeline")
parser.add_argument(
    "--pretrained_dir", default="./runs_organ/run10/", type=str, help="pretrained checkpoint directory"
)
parser.add_argument("--data_dir", default="./amos22/", type=str, help="dataset directory")
parser.add_argument("--json_list", default="dataset_internal_val.json", type=str, help="dataset json file")
parser.add_argument(
    "--pretrained_model_name", default="model.pt", type=str, help="pretrained model name"
)
parser.add_argument(
    "--saved_checkpoint", default="ckpt", type=str, help="Supports torchscript or ckpt pretrained checkpoint type"
)
parser.add_argument("--batch_size", default=1, type=int, help="number of batch size")
parser.add_argument("--mlp_dim", default=3072, type=int, help="mlp dimention in ViT encoder")
parser.add_argument("--hidden_size", default=768, type=int, help="hidden size dimention in ViT encoder")
parser.add_argument("--feature_size", default=64, type=int, help="feature size dimention")
parser.add_argument("--infer_overlap", default=0.5, type=float, help="sliding window inference overlap")
parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=16, type=int, help="number of output channels")
parser.add_argument("--num_heads", default=12, type=int, help="number of attention heads in ViT encoder")
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
parser.add_argument("--additional_information", default="organ_classif_early", help="additional information provided to segmentation model")
parser.add_argument("--classification_layer", default=3, type=int, help="Transformer layer for classification")


def calculate_accuracy(args, model, loader):
    accs_per_organ = {}
    missed_per_organ = {}
    for idx, batch in enumerate(loader):
        val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
        # img_name = batch["image_meta_dict"]["filename_or_obj"][0].split("/")[-1]
        # print("Inference on case {}".format(img_name))
        if "organ_classif" in args.additional_information:
            _, class_logits = model(val_inputs, test_mode=False, class_layer=args.classification_layer)
        else:
            print("Script can only be used for joint organ classification and segmentation models.")
            return
        class_logits = class_logits.cpu()
        class_labels = torch.zeros((class_logits.shape[0], class_logits.shape[1]+1, 1))
        for i in range(class_logits.shape[0]):
            class_labels[i, torch.unique(val_labels[i]).as_tensor().to(int)] = 1
        class_labels = class_labels[:, 1:, :]
        for i in range(class_logits.shape[0]):
            for organ in range(1, 16):
                y_pred = class_logits[i, organ-1, 0].item() > 0
                y_true = class_labels[i, organ-1, 0].item()
                if y_pred == y_true:
                    accs_per_organ.setdefault(organ, [0])[0] += 1
                if y_true == 1:
                    missed_per_organ.setdefault(organ, [0, 0])[1] += 1
                    if y_pred == False:
                        missed_per_organ[organ][0] += 1
        print("{}/{} validation images processed".format(idx+1, len(loader)))
    # Calculate mean score per organ and overall
    total_acc = 0
    total_missed = 0
    for organ in accs_per_organ:
        accs_per_organ[organ] = accs_per_organ[organ][0]/(len(loader)*class_logits.shape[0])
        total_acc += accs_per_organ[organ]
    for organ in missed_per_organ:
        missed_per_organ[organ] = missed_per_organ[organ][0]/missed_per_organ[organ][1]
        total_missed += missed_per_organ[organ]
    mean_acc = total_acc/len(accs_per_organ)
    mean_missed = total_missed/len(missed_per_organ)
    print("Overall classification accuracy: {}".format(mean_acc))
    print("Overall missed predictions: {}".format(mean_missed))

    return mean_acc, accs_per_organ, mean_missed, missed_per_organ


def main():
    args = parser.parse_args()
    args.test_mode = False
    if args.preprocessing == 2:
        loader = get_loader_2(args)
    train_loader = loader[0]
    pretrained_dir = args.pretrained_dir
    model_name = args.pretrained_model_name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_pth = os.path.join(pretrained_dir, model_name)
    if args.saved_checkpoint == "torchscript":
        model = torch.jit.load(pretrained_pth)
    elif args.saved_checkpoint == "ckpt":
        if args.additional_information == "organ_classif":
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
        else:
            print("Script can only be used for joint organ classification and segmentation models.")
            return
        model_dict = torch.load(pretrained_pth)
        model.load_state_dict(model_dict["state_dict"])
    model.eval()
    model.to(device)

    with torch.no_grad():
        calculate_accuracy(args, model, train_loader)


if __name__ == "__main__":
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

    set_determinism()

    main()
