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

### Demo code for final presentation

import argparse
import os

import numpy as np
import torch
from networks.unetr_2d import UNETR_2D
from networks.unetr_2d_organ import UNETR_2D_organ
from data_utils.data_loader_2 import get_loader_2
from data_utils.visualise_data import plot_save_predictions2

from monai.inferers import sliding_window_inference
from monai.utils.misc import set_determinism


parser = argparse.ArgumentParser(description="UNETR segmentation pipeline")
parser.add_argument("--data_dir", default="./amos22/", type=str, help="dataset directory")
parser.add_argument("--json_list", default="dataset_small.json", type=str, help="dataset json file")
parser.add_argument("--mlp_dim", default=3072, type=int, help="mlp dimention in ViT encoder")
parser.add_argument("--hidden_size", default=768, type=int, help="hidden size dimention in ViT encoder")
parser.add_argument("--feature_size", default=64, type=int, help="feature size dimention")
parser.add_argument("--infer_overlap", default=0.5, type=float, help="sliding window inference overlap")
parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=16, type=int, help="number of output channels")
parser.add_argument("--num_heads", default=12, type=int, help="number of attention heads in ViT encoder")
parser.add_argument("--res_block", action="store_true", help="use residual blocks")
parser.add_argument("--conv_block", action="store_true", help="use conv blocks")
parser.add_argument("--space_x", default=1.5, type=float, help="spacing in x direction")
parser.add_argument("--space_y", default=1.5, type=float, help="spacing in y direction")
parser.add_argument("--space_z", default=2.0, type=float, help="spacing in z direction")
parser.add_argument("--roi_x", default=112, type=int, help="roi size in x direction")
parser.add_argument("--roi_y", default=112, type=int, help="roi size in y direction")
parser.add_argument("--roi_z", default=1, type=int, help="roi size in z direction")
parser.add_argument("--dropout_rate", default=0.0, type=float, help="dropout rate")
parser.add_argument("--distributed", action="store_true", help="start distributed training")
parser.add_argument("--workers", default=8, type=int, help="number of workers")
parser.add_argument("--pos_embed", default="perceptron", type=str, help="type of position embedding")
parser.add_argument("--norm_name", default="instance", type=str, help="normalization layer type in decoder")
parser.add_argument("--train_samples", default=40, type=int, help="number of samples per training image")
parser.add_argument("--val_samples", default=1, type=int, help="number of samples per validation image")
parser.add_argument("--train_sampling", default="uniform", type=str, help="sampling distribution of organs during training")
parser.add_argument("--data_augmentation", action="store_true", help="use data augmentation during training")


def visualise_predictions(args, model_best, model_baseline, loader, modality):
    for idx, batch in enumerate(loader):
        val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
        val_inputs_full = torch.cat((val_inputs, val_labels), dim=1)
        val_outputs_best = sliding_window_inference(val_inputs_full, (args.roi_x, args.roi_y), 1, model_best, overlap=args.infer_overlap)
        val_outputs_best = torch.softmax(val_outputs_best, 1).cpu().numpy()
        val_outputs_best = np.argmax(val_outputs_best, axis=1, keepdims=True).astype(np.uint8)
        val_outputs_baseline = sliding_window_inference(val_inputs, (args.roi_x, args.roi_y), 1, model_baseline, overlap=args.infer_overlap)
        val_outputs_baseline = torch.softmax(val_outputs_baseline, 1).cpu().numpy()
        val_outputs_baseline = np.argmax(val_outputs_baseline, axis=1, keepdims=True).astype(np.uint8)
        val_labels = val_labels.cpu().numpy()
        x = val_inputs[0, 0, :, :]
        y_pred_best = val_outputs_best[0, 0, :, :]
        y_pred_baseline = val_outputs_baseline[0, 0, :, :]
        y_true = val_labels[0, 0, :, :]
        plot_save_predictions2(x, y_pred_best, y_pred_baseline, y_true, modality)
        

def main():
    args = parser.parse_args()
    args.test_mode = True
    args.test_type = "test" # "validation" or "test"

    print("Loading test images")
    val_loader = get_loader_2(args)
    loader_ct = val_loader[0]
    loader_mri = val_loader[1]

    print("Loading context-aware model")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_best = UNETR_2D_organ(
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
    model_dict_best = torch.load("./runs_organ/run1b/model.pt")
    model_best.load_state_dict(model_dict_best["state_dict"])
    model_best.eval()
    model_best.to(device)

    print("Loading baseline model")
    model_baseline = UNETR_2D(
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
    model_dict_baseline = torch.load("./runs/rerun18-3/model.pt")
    model_baseline.load_state_dict(model_dict_baseline["state_dict"])
    model_baseline.eval()
    model_baseline.to(device)

    with torch.no_grad():
        print("Generating example predictions for CT scan")
        visualise_predictions(args, model_best, model_baseline, loader_ct, modality="CT")
        print("Generating example predictions for MRI scan")
        visualise_predictions(args, model_best, model_baseline, loader_mri, modality="MRI")


if __name__ == "__main__":
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

    set_determinism()

    main()