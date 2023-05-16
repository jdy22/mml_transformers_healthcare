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

import math
import os

import numpy as np
import torch

from monai import data, transforms
from monai.data import load_decathlon_datalist

from data_utils.data_loader import Sampler


def get_loader_2(args):
    data_dir = args.data_dir
    # datalist_json = os.path.join(data_dir, args.json_list)
    datalist_json = args.json_list

    if args.train_sampling == "uniform":
        train_crop_ratios = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    elif args.train_sampling == "unbalanced":
        train_crop_ratios = [1, 1, 1, 1, 2, 2, 1, 2, 1, 2, 2, 2, 2, 2, 2, 2]

    train_transform_ct = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.AddChanneld(keys=["image", "label"]),
            transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            transforms.Spacingd(
                keys=["image", "label"], pixdim=(args.space_x, args.space_y, args.space_z), mode=("bilinear", "nearest")
            ),
            transforms.ThresholdIntensityd(keys=["image"], threshold=-991, above=False, cval=-991),
            transforms.ThresholdIntensityd(keys=["image"], threshold=362, above=True, cval=362),
            transforms.NormalizeIntensityd(keys=["image"], subtrahend=50, divisor=141),
            transforms.RandCropByLabelClassesd(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                ratios=train_crop_ratios,
                num_classes=16,
                num_samples=args.train_samples,
            ),
            transforms.SqueezeDimd(keys=["image", "label"], dim=-1),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )
    train_transform_mri = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.AddChanneld(keys=["image", "label"]),
            transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            transforms.Spacingd(
                keys=["image", "label"], pixdim=(args.space_x, args.space_y, args.space_z), mode=("bilinear", "nearest")
            ),
            transforms.NormalizeIntensityd(keys=["image"]),
            transforms.RandCropByLabelClassesd(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                ratios=train_crop_ratios,
                num_classes=16,
                num_samples=args.train_samples,
            ),
            transforms.SqueezeDimd(keys=["image", "label"], dim=-1),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )
    val_transform_ct = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.AddChanneld(keys=["image", "label"]),
            transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            transforms.Spacingd(
                keys=["image", "label"], pixdim=(args.space_x, args.space_y, args.space_z), mode=("bilinear", "nearest")
            ),
            transforms.ThresholdIntensityd(keys=["image"], threshold=-991, above=False, cval=-991),
            transforms.ThresholdIntensityd(keys=["image"], threshold=362, above=True, cval=362),
            transforms.NormalizeIntensityd(keys=["image"], subtrahend=50, divisor=141),
            transforms.RandCropByLabelClassesd(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(-1, -1, 1),
                ratios=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                num_classes=16,
                num_samples=args.val_samples,
            ),
            transforms.SqueezeDimd(keys=["image", "label"], dim=-1),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )
    val_transform_mri = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.AddChanneld(keys=["image", "label"]),
            transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            transforms.Spacingd(
                keys=["image", "label"], pixdim=(args.space_x, args.space_y, args.space_z), mode=("bilinear", "nearest")
            ),
            transforms.NormalizeIntensityd(keys=["image"]),
            transforms.RandCropByLabelClassesd(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(-1, -1, 1),
                ratios=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                num_classes=16,
                num_samples=args.val_samples,
            ),
            transforms.SqueezeDimd(keys=["image", "label"], dim=-1),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )
    
    if args.test_mode:
        test_files_ct = load_decathlon_datalist(datalist_json, True, "internal-validation-ct", base_dir=data_dir)
        test_ds_ct = data.Dataset(data=test_files_ct, transform=val_transform_ct)
        test_sampler_ct = Sampler(test_ds_ct, shuffle=False) if args.distributed else None
        test_loader_ct = data.DataLoader(
            test_ds_ct,
            batch_size=1,
            shuffle=False,
            num_workers=args.workers,
            sampler=test_sampler_ct,
            pin_memory=True,
            persistent_workers=True,
        )
        test_files_mri = load_decathlon_datalist(datalist_json, True, "internal-validation-mri", base_dir=data_dir)
        test_ds_mri = data.Dataset(data=test_files_mri, transform=val_transform_mri)
        test_sampler_mri = Sampler(test_ds_mri, shuffle=False) if args.distributed else None
        test_loader_mri = data.DataLoader(
            test_ds_mri,
            batch_size=1,
            shuffle=False,
            num_workers=args.workers,
            sampler=test_sampler_mri,
            pin_memory=True,
            persistent_workers=True,
        )
        loader = [test_loader_ct, test_loader_mri]
    else:
        train_files_ct = load_decathlon_datalist(datalist_json, True, "training-ct", base_dir=data_dir)
        train_files_mri = load_decathlon_datalist(datalist_json, True, "training-mri", base_dir=data_dir)
        if args.use_normal_dataset:
            train_ds_ct = data.Dataset(data=train_files_ct, transform=train_transform_ct)
            train_ds_mri = data.Dataset(data=train_files_mri, transform=train_transform_mri)
        else:
            train_ds_ct = data.CacheDataset(
                data=train_files_ct, transform=train_transform_ct, cache_num=24, cache_rate=1.0, num_workers=args.workers
            )
            train_ds_mri = data.CacheDataset(
                data=train_files_mri, transform=train_transform_mri, cache_num=24, cache_rate=1.0, num_workers=args.workers
            )
        train_ds_full = torch.utils.data.ConcatDataset([train_ds_ct, train_ds_mri])
        train_sampler = Sampler(train_ds_full) if args.distributed else None
        train_loader = data.DataLoader(
            train_ds_full,
            batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            num_workers=args.workers,
            sampler=train_sampler,
            pin_memory=True,
            persistent_workers=True,
        )
        val_files_ct = load_decathlon_datalist(datalist_json, True, "internal-validation-ct", base_dir=data_dir)
        val_files_mri = load_decathlon_datalist(datalist_json, True, "internal-validation-mri", base_dir=data_dir)
        val_ds_ct = data.Dataset(data=val_files_ct, transform=val_transform_ct)
        val_ds_mri = data.Dataset(data=val_files_mri, transform=val_transform_mri)
        val_ds_full = torch.utils.data.ConcatDataset([val_ds_ct, val_ds_mri])
        val_sampler = Sampler(val_ds_full, shuffle=False) if args.distributed else None
        val_loader = data.DataLoader(
            val_ds_full,
            batch_size=1,
            shuffle=False,
            num_workers=args.workers,
            sampler=val_sampler,
            pin_memory=True,
            persistent_workers=True,
        )
        loader = [train_loader, val_loader]

    return loader