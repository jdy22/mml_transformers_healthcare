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

### Functions for preprocessing pipeline 1
# Adapted from the original UNETR code

import math
import os

import numpy as np
import torch

from monai import data, transforms
from monai.data import load_decathlon_datalist


class Sampler(torch.utils.data.Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, make_even=True):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.shuffle = shuffle
        self.make_even = make_even
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        indices = list(range(len(self.dataset)))
        self.valid_length = len(indices[self.rank : self.total_size : self.num_replicas])

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))
        if self.make_even:
            if len(indices) < self.total_size:
                if self.total_size - len(indices) < len(indices):
                    indices += indices[: (self.total_size - len(indices))]
                else:
                    extra_ids = np.random.randint(low=0, high=len(indices), size=self.total_size - len(indices))
                    indices += [indices[ids] for ids in extra_ids]
            assert len(indices) == self.total_size
        indices = indices[self.rank : self.total_size : self.num_replicas]
        self.num_samples = len(indices)
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


def get_loader(args):
    data_dir = args.data_dir
    # datalist_json = os.path.join(data_dir, args.json_list)
    datalist_json = args.json_list

    if args.train_sampling == "uniform":
        train_crop_ratios = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    elif args.train_sampling == "unbalanced":
        train_crop_ratios = [1, 1, 1, 1, 2, 2, 1, 2, 1, 2, 2, 2, 2, 2, 2, 2]

    if args.data_augmentation:
        train_transform = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image", "label"]),
                transforms.AddChanneld(keys=["image", "label"]),
                transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
                transforms.Spacingd(
                    keys=["image", "label"], pixdim=(args.space_x, args.space_y, args.space_z), mode=("bilinear", "nearest")
                ),
                transforms.ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=args.lower, upper=args.upper, b_min=0, b_max=1, clip=True
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
                transforms.RandRotated(keys=["image", "label"], range_x=0.52, prob=0.2, mode=("bilinear", "nearest")),
                transforms.RandScaleIntensityd(keys=["image"], factors=(-0.3, 0.4), prob=0.2),
                transforms.RandGaussianNoised(keys=["image"], prob=0.1, mean=0.0, std=0.1),
                transforms.RandAdjustContrastd(keys=["image"], prob=0.15, gamma=(0.75, 1.5)),
                transforms.SqueezeDimd(keys=["image", "label"], dim=-1),
                transforms.ToTensord(keys=["image", "label"]),
            ]
        )
    else:
        train_transform = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image", "label"]),
                transforms.AddChanneld(keys=["image", "label"]),
                transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
                transforms.Spacingd(
                    keys=["image", "label"], pixdim=(args.space_x, args.space_y, args.space_z), mode=("bilinear", "nearest")
                ),
                transforms.ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=args.lower, upper=args.upper, b_min=0, b_max=1, clip=True
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
    val_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.AddChanneld(keys=["image", "label"]),
            transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            transforms.Spacingd(
                keys=["image", "label"], pixdim=(args.space_x, args.space_y, args.space_z), mode=("bilinear", "nearest")
            ),
            transforms.ScaleIntensityRangePercentilesd(
                keys=["image"], lower=args.lower, upper=args.upper, b_min=0, b_max=1, clip=True
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
        if args.test_type == "validation":
            test_files_ct = load_decathlon_datalist(datalist_json, True, "internal-validation-ct", base_dir=data_dir)
        elif args.test_type == "test":
            test_files_ct = load_decathlon_datalist(datalist_json, True, "validation-ct", base_dir=data_dir)
        test_ds_ct = data.Dataset(data=test_files_ct, transform=val_transform)
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
        if args.test_type == "validation":
            test_files_mri = load_decathlon_datalist(datalist_json, True, "internal-validation-mri", base_dir=data_dir)
        elif args.test_type == "test":
            test_files_mri = load_decathlon_datalist(datalist_json, True, "validation-mri", base_dir=data_dir)
        test_ds_mri = data.Dataset(data=test_files_mri, transform=val_transform)
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
        datalist = load_decathlon_datalist(datalist_json, True, "training", base_dir=data_dir)
        if args.use_normal_dataset:
            train_ds = data.Dataset(data=datalist, transform=train_transform)
        else:
            train_ds = data.CacheDataset(
                data=datalist, transform=train_transform, cache_num=24, cache_rate=1.0, num_workers=args.workers
            )
        train_sampler = Sampler(train_ds) if args.distributed else None
        train_loader = data.DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            num_workers=args.workers,
            sampler=train_sampler,
            pin_memory=True,
            persistent_workers=True,
        )
        val_files = load_decathlon_datalist(datalist_json, True, "internal-validation", base_dir=data_dir)
        val_ds = data.Dataset(data=val_files, transform=val_transform)
        val_sampler = Sampler(val_ds, shuffle=False) if args.distributed else None
        val_loader = data.DataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=args.workers,
            sampler=val_sampler,
            pin_memory=True,
            persistent_workers=True,
        )
        loader = [train_loader, val_loader]

    return loader