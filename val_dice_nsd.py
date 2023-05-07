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
from trainer import dice
from data_utils.data_loader import get_loader

from monai.inferers import sliding_window_inference
from monai.metrics import compute_surface_dice

parser = argparse.ArgumentParser(description="UNETR segmentation pipeline")
parser.add_argument(
    "--pretrained_dir", default="./runs/run3/", type=str, help="pretrained checkpoint directory"
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
parser.add_argument("--feature_size", default=16, type=int, help="feature size dimention")
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
parser.add_argument("--roi_x", default=80, type=int, help="roi size in x direction")
parser.add_argument("--roi_y", default=80, type=int, help="roi size in y direction")
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
parser.add_argument("--lower", default=1.0, type=float, help="lower percentile in ScaleIntensityRangePercentilesd")
parser.add_argument("--upper", default=99.0, type=float, help="upper percentile in ScaleIntensityRangePercentilesd")
parser.add_argument("--train_samples", default=40, type=int, help="number of samples per training image")
parser.add_argument("--val_samples", default=20, type=int, help="number of samples per validation image")
parser.add_argument("--nsd_threshold", default=3, type=int, help="class_thresholds in compute_surface_dice")


def calculate_score(metric, args, model, loader):
    # Options for metric: "dice" or "nsd"
    scores_per_organ = {}
    nan_inf_count = 0
    for idx, batch in enumerate(loader):
        val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
        # img_name = batch["image_meta_dict"]["filename_or_obj"][0].split("/")[-1]
        # print("Inference on case {}".format(img_name))
        val_outputs = sliding_window_inference(val_inputs, (args.roi_x, args.roi_y), 1, model, overlap=args.infer_overlap)
        val_outputs = torch.softmax(val_outputs, 1).cpu().numpy()
        val_outputs = np.argmax(val_outputs, axis=1, keepdims=True).astype(np.uint8)
        val_labels = val_labels.cpu().numpy()
        for i in range(val_outputs.shape[0]):
            for organ in range(1, 16):
                y_pred = (val_outputs[i] == organ)
                y_true = (val_labels[i] == organ)
                # Skip if organ does not exist in true labels
                if np.sum(np.sum(np.sum(y_true))) == 0:
                    continue
                if metric == "dice":
                    score = dice(y_pred, y_true)
                elif metric == "nsd":
                    y_pred = np.expand_dims(y_pred, 0)
                    y_true = np.expand_dims(y_true, 0)
                    score = compute_surface_dice(torch.Tensor(y_pred), torch.Tensor(y_true), [args.nsd_threshold])[0, 0]
                    if np.isposinf(score) or np.isnan(score):
                        nan_inf_count += 1
                        continue
                scores_per_organ.setdefault(organ, []).append(score)
        print("{}/{} validation images processed".format(idx+1, len(loader)))
    # Calculate mean score per organ and overall
    total_score = 0
    for organ in scores_per_organ:
        scores_per_organ[organ] = np.mean(scores_per_organ[organ])
        total_score += scores_per_organ[organ]
    mean_score_overall = total_score/len(scores_per_organ)
    print("Overall mean {} score: {}".format(metric, mean_score_overall))
    print("Number of nan and inf values = {}".format(nan_inf_count))

    return mean_score_overall, scores_per_organ


def main():
    args = parser.parse_args()
    args.test_mode = True
    val_loader = get_loader(args)
    loader_ct = val_loader[0]
    loader_mri = val_loader[1]
    pretrained_dir = args.pretrained_dir
    model_name = args.pretrained_model_name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_pth = os.path.join(pretrained_dir, model_name)
    if args.saved_checkpoint == "torchscript":
        model = torch.jit.load(pretrained_pth)
    elif args.saved_checkpoint == "ckpt":
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
        mean_dice_ct, mean_dice_per_organ_ct = calculate_score("dice", args, model, loader_ct)
        mean_nsd_ct, mean_nsd_per_organ_ct = calculate_score("nsd", args, model, loader_ct)
        
        mean_dice_mri, mean_dice_per_organ_mri = calculate_score("dice", args, model, loader_mri)
        mean_nsd_mri, mean_nsd_per_organ_mri = calculate_score("nsd", args, model, loader_mri)

        print("Final scores:")
        print(f"CT: mDice = {mean_dice_ct}, mNSD = {mean_nsd_ct}")
        print(f"MRI: mDice = {mean_dice_mri}, mNSD = {mean_nsd_mri}")

        print(mean_dice_per_organ_ct)
        print(mean_nsd_per_organ_ct)
        print(mean_dice_per_organ_mri)
        print(mean_nsd_per_organ_mri)


if __name__ == "__main__":
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

    main()
