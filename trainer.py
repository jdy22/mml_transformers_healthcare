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

import os
import shutil
import time
import json

import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data.distributed
from tensorboardX import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from monai_research_contributions_main.UNETR.BTCV.utils.utils import distributed_all_gather

from monai.data import decollate_batch


def dice(x, y):
    intersect = np.sum(np.sum(np.sum(np.sum(x * y))))
    y_sum = np.sum(np.sum(np.sum(np.sum(y))))
    if y_sum == 0:
        return 0.0
    x_sum = np.sum(np.sum(np.sum(np.sum(x))))
    return 2 * intersect / (x_sum + y_sum)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = np.where(self.count > 0, self.sum / self.count, self.sum)


def train_epoch(model, loader, optimizer, scaler, epoch, loss_func, args):
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()
    if args.additional_information == "organ_classif":
        run_seg_loss = AverageMeter()
        run_class_loss = AverageMeter()
    for idx, batch_data in enumerate(loader):
        if isinstance(batch_data, list):
            data, target = batch_data
        else:
            data, target, filename = batch_data["image"], batch_data["label"], batch_data["image_meta_dict"]["filename_or_obj"][0]
        data, target = data.cuda(args.rank), target.cuda(args.rank)
        load_time = time.time() - start_time
        inter_time = time.time()
        for param in model.parameters():
            param.grad = None
        with autocast(enabled=args.amp):
            image_index = int(filename[-10:-7])
            if image_index < 500:
                modality = "CT"
            else:
                modality = "MRI"

            if args.additional_information == "modality_concat":
                logits = model(data, modality, info_mode="concat")
            elif args.additional_information == "modality_concat2":
                logits = model(data, modality, info_mode="concat2")
            elif args.additional_information == "modality_add":
                logits = model(data, modality, info_mode="add")
            elif args.additional_information == "organ":
                data_in = torch.cat((data, target), dim=1)
                logits = model(data_in)
            elif args.additional_information == "organ_classif":
                seg_logits, class_logits = model(data, test_mode=False)
            else:
                logits = model(data)

            if args.additional_information == "organ_classif":
                seg_loss = loss_func[0](seg_logits, target)
                class_labels = torch.zeros((class_logits.shape[0], class_logits.shape[1]+1, 1))
                for i in range(class_logits.shape[0]):
                    class_labels[i, torch.unique(target[i]).as_tensor().to(int)] = 1
                class_labels = class_labels[:, 1:, :].cuda(args.rank)
                class_loss = loss_func[1](class_logits, class_labels)
                loss = seg_loss + args.loss_combination_factor*class_loss
            else:
                loss = loss_func(logits, target)
        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        if args.distributed:
            loss_list = distributed_all_gather([loss], out_numpy=True, is_valid=idx < loader.sampler.valid_length)
            run_loss.update(
                np.mean(np.mean(np.stack(loss_list, axis=0), axis=0), axis=0), n=args.batch_size * args.world_size
            )
        else:
            run_loss.update(loss.item(), n=args.batch_size)
            if args.additional_information == "organ_classif":
                run_seg_loss.update(seg_loss.item(), n=args.batch_size)
                run_class_loss.update(class_loss.item(), n=args.batch_size)
        if args.rank == 0:
            if args.additional_information == "organ_classif":
                print(
                    "Epoch {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(loader)),
                    "total loss: {:.4f}".format(run_loss.avg),
                    "seg loss: {:.4f}".format(run_seg_loss.avg),
                    "class loss: {:.4f}".format(run_class_loss.avg),
                    "time {:.2f}s".format(time.time() - start_time),
                    "loader time {:.2f}s".format(load_time),
                    "optimisation time {:.2f}s".format(time.time() - inter_time),
                )            
            else:
                print(
                    "Epoch {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(loader)),
                    "loss: {:.4f}".format(run_loss.avg),
                    "time {:.2f}s".format(time.time() - start_time),
                    "loader time {:.2f}s".format(load_time),
                    "optimisation time {:.2f}s".format(time.time() - inter_time),
                )
        start_time = time.time()
    for param in model.parameters():
        param.grad = None
    return run_loss.avg


def val_epoch(model, loader, epoch, acc_func, args, model_inferer=None, post_label=None, post_pred=None):
    model.eval()
    start_time = time.time()
    with torch.no_grad():
        accs = []
        for idx, batch_data in enumerate(loader):
            if isinstance(batch_data, list):
                data, target = batch_data
            else:
                data, target, filename = batch_data["image"], batch_data["label"], batch_data["image_meta_dict"]["filename_or_obj"][0]
            data, target = data.cuda(args.rank), target.cuda(args.rank)
            with autocast(enabled=args.amp):
                image_index = int(filename[-10:-7])
                if image_index < 500:
                    modality = "CT"
                else:
                    modality = "MRI"
                if model_inferer is not None:
                    if args.additional_information == "modality_concat": 
                        logits = model_inferer(data, modality=modality, info_mode="concat")
                    elif args.additional_information == "modality_concat2": 
                        logits = model_inferer(data, modality=modality, info_mode="concat2")
                    elif args.additional_information == "modality_add":
                        logits = model_inferer(data, modality=modality, info_mode="add")
                    elif args.additional_information == "organ":
                        data_in = torch.cat((data, target), dim=1)
                        logits = model_inferer(data_in)
                    elif args.additional_information == "organ_classif":
                        logits = model_inferer(data, test_mode=True)
                    else:
                        logits = model_inferer(data)
                else:
                    logits = model(data)
            if not logits.is_cuda:
                target = target.cpu()
            val_labels_list = decollate_batch(target)
            val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
            val_outputs_list = decollate_batch(logits)
            val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
            acc = acc_func(y_pred=val_output_convert, y=val_labels_convert)
            acc = acc.cuda(args.rank)
            if args.distributed:
                acc_list = distributed_all_gather([acc], out_numpy=True, is_valid=idx < loader.sampler.valid_length)
                avg_acc = np.mean([np.nanmean(l) for l in acc_list])
            else:
                acc_list = acc.detach().cpu().numpy()
                avg_acc = np.mean([np.nanmean(l) for l in acc_list])
            accs.append(avg_acc)
            if args.rank == 0:
                print(
                    "Val {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(loader)),
                    "acc",
                    avg_acc,
                    "time {:.2f}s".format(time.time() - start_time),
                )
            start_time = time.time()
    return sum(accs)/len(accs)


def save_checkpoint(model, epoch, args, filename="model.pt", best_acc=0, optimizer=None, scheduler=None):
    state_dict = model.state_dict() if not args.distributed else model.module.state_dict()
    save_dict = {"epoch": epoch, "best_acc": best_acc, "state_dict": state_dict}
    if optimizer is not None:
        save_dict["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        save_dict["scheduler"] = scheduler.state_dict()
    filename = os.path.join(args.logdir, filename)
    torch.save(save_dict, filename)
    print("Saving checkpoint", filename)


def run_training(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_func,
    acc_func,
    args,
    model_inferer=None,
    scheduler=None,
    start_epoch=0,
    post_label=None,
    post_pred=None,
):
    writer = None
    if args.logdir is not None and args.rank == 0:
        writer = SummaryWriter(log_dir=args.logdir)
        if args.rank == 0:
            print("Writing Tensorboard logs to ", args.logdir)
        # Write out args to txt file
        with open(args.logdir + "/args.txt", "w") as file:
            json.dump(args.__dict__, file, indent=2)
    scaler = None
    if args.amp:
        scaler = GradScaler()
    val_acc_max = 0.0
    for epoch in range(start_epoch, args.max_epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
            torch.distributed.barrier()
        print(args.rank, time.ctime(), "Epoch:", epoch)
        epoch_time = time.time()
        train_loss = train_epoch(
            model, train_loader, optimizer, scaler=scaler, epoch=epoch, loss_func=loss_func, args=args
        )
        if args.rank == 0:
            print(
                "Final training  {}/{}".format(epoch, args.max_epochs - 1),
                "loss: {:.4f}".format(train_loss),
                "time {:.2f}s".format(time.time() - epoch_time),
            )
        if args.rank == 0 and writer is not None:
            writer.add_scalar("train_loss", train_loss, epoch)
        b_new_best = False
        if (epoch + 1) % args.val_every == 0:
            if args.distributed:
                torch.distributed.barrier()
            epoch_time = time.time()
            val_avg_acc = val_epoch(
                model,
                val_loader,
                epoch=epoch,
                acc_func=acc_func,
                model_inferer=model_inferer,
                args=args,
                post_label=post_label,
                post_pred=post_pred,
            )
            if args.rank == 0:
                print(
                    "Final validation  {}/{}".format(epoch, args.max_epochs - 1),
                    "acc",
                    val_avg_acc,
                    "time {:.2f}s".format(time.time() - epoch_time),
                )
                if writer is not None:
                    writer.add_scalar("val_acc", val_avg_acc, epoch)
                if val_avg_acc > val_acc_max:
                    print("new best ({:.6f} --> {:.6f}). ".format(val_acc_max, val_avg_acc))
                    val_acc_max = val_avg_acc
                    b_new_best = True
                    if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
                        save_checkpoint(
                            model, epoch, args, best_acc=val_acc_max, optimizer=optimizer, scheduler=scheduler
                        )
            # if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
            #     save_checkpoint(model, epoch, args, best_acc=val_acc_max, filename="model_final.pt")
            #     if b_new_best:
            #         print("Copying to model.pt new best model!!!!")
            #         shutil.copyfile(os.path.join(args.logdir, "model_final.pt"), os.path.join(args.logdir, "model.pt"))

        if scheduler is not None:
            scheduler.step()

    print("Training Finished !, Best Accuracy: ", val_acc_max)

    return val_acc_max
