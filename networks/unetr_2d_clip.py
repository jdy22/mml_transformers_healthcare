# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

### Model architecture of context-aware 2D UNETR models which use CLIP embeddings
# Adapted from the original UNETR code

from typing import Tuple, Union

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle

from monai.networks.blocks.patchembedding import PatchEmbeddingBlock
from monai.networks.blocks.transformerblock import TransformerBlock
from monai.networks.blocks import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.nets import ViT

from networks.unetr_2d_organ import add_organ_info


with open("clip_embeddings.pk1", "rb") as target:
    clip_embeddings = pickle.load(target)
ct_pos_embeddings = clip_embeddings[0]
ct_neg_embeddings = clip_embeddings[1]
mri_pos_embeddings = clip_embeddings[2]
mri_neg_embeddings = clip_embeddings[3]


def get_organ_info(labels, organ_tokens):
    """
    labels : To indicate which organs are present
    organ_tokens : Dicts with relevant tokens to add (CLIP embeddings)
    Information is concatenated and then global average pooling is applied
    """
    batch_size = labels.shape[0]
    organ_info = []
    avg_pool = nn.AdaptiveAvgPool1d(1)
    organs_present = torch.unique(labels[0])
    embeddings = []
    for organ in range(1, 16):
        if organ in organs_present:
            organ_embedding = torch.squeeze(organ_tokens[str(organ)])
            organ_embedding = torch.unsqueeze(organ_embedding, dim=0)
            organ_embedding = torch.unsqueeze(organ_embedding, dim=-1)
            embeddings.append(organ_embedding)
    if len(embeddings) == 0:
        organ_embedding = torch.squeeze(organ_tokens["0"])
        organ_embedding = torch.unsqueeze(organ_embedding, dim=0)
        organ_embedding = torch.unsqueeze(organ_embedding, dim=-1)
        embeddings.append(organ_embedding)
    organ_info.append(avg_pool(torch.cat(embeddings, dim=-1)))
    for i in range(1, batch_size):
        organs_present = torch.unique(labels[i])
        embeddings = []
        for organ in range(1, 16):
            if organ in organs_present:
                organ_embedding = torch.squeeze(organ_tokens[str(organ)])
                organ_embedding = torch.unsqueeze(organ_embedding, dim=0)
                organ_embedding = torch.unsqueeze(organ_embedding, dim=-1)
                embeddings.append(organ_embedding)
        if len(embeddings) == 0:
            organ_embedding = torch.squeeze(organ_tokens["0"])
            organ_embedding = torch.unsqueeze(organ_embedding, dim=0)
            organ_embedding = torch.unsqueeze(organ_embedding, dim=-1)
            embeddings.append(organ_embedding)
        organ_info.append(avg_pool(torch.cat(embeddings, dim=-1)))
    organ_info = torch.cat(organ_info, dim=0)
    return organ_info


class ViT_clip(nn.Module):
    """
    Vision Transformer (ViT), based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"
    ViT supports Torchscript but only works for Pytorch after 1.8.
    """

    def __init__(
        self,
        in_channels: int,
        img_size: Sequence[int] | int,
        patch_size: Sequence[int] | int,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_layers: int = 12,
        num_heads: int = 12,
        pos_embed: str = "conv",
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
        post_activation="Tanh",
        qkv_bias: bool = False,
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            img_size: dimension of input image.
            patch_size: dimension of patch size.
            hidden_size: dimension of hidden layer.
            mlp_dim: dimension of feedforward layer.
            num_layers: number of transformer blocks.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            dropout_rate: faction of the input units to drop.
            spatial_dims: number of spatial dimensions.
            post_activation: add a final acivation function to the classification head when `classification` is True.
                Default to "Tanh" for `nn.Tanh()`. Set to other values to remove this function.
            qkv_bias: apply bias to the qkv linear layer in self attention block
        Examples::
            # for single channel input with image size of (96,96,96), conv position embedding and segmentation backbone
            >>> net = ViT(in_channels=1, img_size=(96,96,96), pos_embed='conv')
        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.patch_embedding = PatchEmbeddingBlock(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            pos_embed=pos_embed,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
        )
        self.blocks = nn.ModuleList(
            [TransformerBlock(hidden_size, mlp_dim, num_heads, dropout_rate, qkv_bias) for i in range(num_layers)]
        )
        self.norm = nn.LayerNorm(hidden_size)

        self.organ_tokens_CT = {k: v.to(device="cuda:0") for k, v in ct_pos_embeddings.items()}
        self.no_organ_tokens_CT = {k: v.to(device="cuda:0") for k, v in ct_neg_embeddings.items()}
        self.organ_tokens_MRI = {k: v.to(device="cuda:0") for k, v in mri_pos_embeddings.items()}
        self.no_organ_tokens_MRI = {k: v.to(device="cuda:0") for k, v in mri_neg_embeddings.items()}

        # For testing only
        # self.organ_tokens_CT = ct_pos_embeddings
        # self.no_organ_tokens_CT = ct_neg_embeddings
        # self.organ_tokens_MRI = mri_pos_embeddings
        # self.no_organ_tokens_MRI = mri_neg_embeddings


    def forward(self, x_in, modality):
        # Options for modality: "CT" or "MRI"
        # First channel of x_in: image
        # Second channel of x_in: labels
        x = x_in[:, None, 0, :, :]
        labels = x_in[:, 1, :, :]

        x = self.patch_embedding(x)
        n_patches = x.shape[1]

        if modality == "CT":
            x_full = add_organ_info(x, labels, self.organ_tokens_CT, self.no_organ_tokens_CT)
        elif modality == "MRI":
            x_full = add_organ_info(x, labels, self.organ_tokens_MRI, self.no_organ_tokens_MRI)
 
        hidden_states_out = []
        for blk in self.blocks:
            x_full = blk(x_full)
            hidden_states_out.append(x_full[:, :n_patches, :])
        x_full = self.norm(x_full)
        x_out = x_full[:, :n_patches, :]

        return x_out, hidden_states_out                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       


class UNETR_2D_clip(nn.Module):
    """
    UNETR based on: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        img_size: Tuple[int, int],
        feature_size: int = 16,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_heads: int = 12,
        pos_embed: str = "perceptron",
        norm_name: Union[Tuple, str] = "instance",
        conv_block: bool = False,
        res_block: bool = True,
        dropout_rate: float = 0.0,
        info_mode: str = "late",
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            img_size: dimension of input image.
            feature_size: dimension of network feature size.
            hidden_size: dimension of hidden layer.
            mlp_dim: dimension of feedforward layer.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            norm_name: feature normalization type and arguments.
            conv_block: bool argument to determine if convolutional block is used.
            res_block: bool argument to determine if residual block is used.
            dropout_rate: faction of the input units to drop.
            info_mode: early, late.

        Examples::

            # for single channel input 4-channel output with patch size of (96,96,96), feature size of 32 and batch norm
            >>> net = UNETR(in_channels=1, out_channels=4, img_size=(96,96,96), feature_size=32, norm_name='batch')

            # for 4-channel input 3-channel output with patch size of (128,128,128), conv position embedding and instance norm
            >>> net = UNETR(in_channels=4, out_channels=3, img_size=(128,128,128), pos_embed='conv', norm_name='instance')

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise AssertionError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise AssertionError("hidden size should be divisible by num_heads.")

        if pos_embed not in ["conv", "perceptron"]:
            raise KeyError(f"Position embedding layer of type {pos_embed} is not supported.")

        self.num_layers = 12
        self.patch_size = (16, 16)
        self.feat_size = (
            img_size[0] // self.patch_size[0],
            img_size[1] // self.patch_size[1],
        )
        self.hidden_size = hidden_size
        self.info_mode = info_mode

        if self.info_mode == "early":
            self.vit = ViT_clip(
                in_channels=in_channels,
                img_size=img_size,
                patch_size=self.patch_size,
                hidden_size=hidden_size,
                mlp_dim=mlp_dim,
                num_layers=self.num_layers,
                num_heads=num_heads,
                pos_embed=pos_embed,
                dropout_rate=dropout_rate,
                spatial_dims=2,
            )
        elif self.info_mode == "late":
            self.vit = ViT(
                in_channels=in_channels,
                img_size=img_size,
                patch_size=self.patch_size,
                hidden_size=hidden_size,
                mlp_dim=mlp_dim,
                num_layers=self.num_layers,
                num_heads=num_heads,
                pos_embed=pos_embed,
                classification=False,
                dropout_rate=dropout_rate,
                spatial_dims=2,
            )

        self.encoder1 = UnetrBasicBlock(
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UnetrPrUpBlock(
            spatial_dims=2,
            in_channels=hidden_size,
            out_channels=feature_size * 2,
            num_layer=2,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder3 = UnetrPrUpBlock(
            spatial_dims=2,
            in_channels=hidden_size,
            out_channels=feature_size * 4,
            num_layer=1,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder4 = UnetrPrUpBlock(
            spatial_dims=2,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            num_layer=0,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.decoder5 = UnetrUpBlock(
            spatial_dims=2,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=2,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=2,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=2,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )

        if self.info_mode == "early":
            self.out = UnetOutBlock(spatial_dims=2, in_channels=feature_size, out_channels=out_channels)
        elif self.info_mode == "late":
            self.organ_tokens_CT = {k: v.to(device="cuda:0") for k, v in ct_pos_embeddings.items()}
            self.organ_tokens_MRI = {k: v.to(device="cuda:0") for k, v in mri_pos_embeddings.items()}
            self.pool_image = nn.Sequential(
                nn.GroupNorm(16, self.hidden_size),
                nn.ReLU(inplace=True),
                torch.nn.AdaptiveAvgPool2d((1,1)),
            )
            self.controller = nn.Conv1d(2*self.hidden_size, 816, kernel_size=1, stride=1, padding=0)
            self.pre_out = nn.Sequential(
                nn.GroupNorm(16, feature_size),
                nn.ReLU(inplace=True),
                nn.Conv2d(feature_size, 16, kernel_size=1)
            )


    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], hidden_size)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x
    

    def segmentor(self, x, params):
        # x shape = [1, 16, 112, 112]
        # params shape = [816]
        conv1_w = torch.reshape(params[:256], (16, 16, 1, 1))
        conv1_b = params[256:272]
        conv2_w = torch.reshape(params[272:528], (16, 16, 1, 1))
        conv2_b = params[528:544]
        conv3_w = torch.reshape(params[544:800], (16, 16, 1, 1))
        conv3_b = params[800:]
        x2 = F.conv2d(x, weight=conv1_w, bias=conv1_b)
        x2 = F.relu(x2)
        x3 = F.conv2d(x2, weight=conv2_w, bias=conv2_b)
        x3 = F.relu(x3)
        out = F.conv2d(x3, weight=conv3_w, bias=conv3_b)
        return out
    

    def forward(self, x_in, modality):
        if self.info_mode == "early":
            x, hidden_states_out = self.vit(x_in, modality)
        elif self.info_mode == "late":
            x = x_in[:, None, 0, :, :]
            labels = x_in[:, 1, :, :]
            x, hidden_states_out = self.vit(x)

        enc1 = self.encoder1(x_in[:, None, 0, :, :])
        x2 = hidden_states_out[2]
        enc2 = self.encoder2(self.proj_feat(x2, self.hidden_size, self.feat_size))
        x3 = hidden_states_out[5]
        enc3 = self.encoder3(self.proj_feat(x3, self.hidden_size, self.feat_size))
        x4 = hidden_states_out[8]
        enc4 = self.encoder4(self.proj_feat(x4, self.hidden_size, self.feat_size))
        dec4 = self.proj_feat(x, self.hidden_size, self.feat_size)
        dec3 = self.decoder5(dec4, enc4)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        out = self.decoder2(dec1, enc1) # shape = [40, 64, 112, 112]

        if self.info_mode == "early":
            logits = self.out(out)
        elif self.info_mode == "late":
            image_feat = self.pool_image(dec4) # shape = [40, 768, 1, 1]
            if modality == "CT":
                organ_feat = get_organ_info(labels, self.organ_tokens_CT) # shape = [40, 768, 1]
            elif modality == "MRI":
                organ_feat = get_organ_info(labels, self.organ_tokens_MRI) # shape = [40, 768, 1]
            image_feat = torch.squeeze(image_feat, dim=-1) # shape = [40, 768, 1]
            controller_in = torch.cat((image_feat, organ_feat), dim=1) # shape = [40, 1536, 1]
            segmentor_params = self.controller(controller_in) # shape = [40, 816, 1]
            segmentor_in = self.pre_out(out) # shape = [40, 16, 112, 112]
            logits = []
            for i in range(segmentor_in.shape[0]):
                x_segmentor = segmentor_in[None, i, :, :, :] # shape = [1, 16, 112, 112]
                params = segmentor_params[i, :, 0] # shape = [816]
                segmentor_out = self.segmentor(x_segmentor, params) # shape = [1, 16, 112, 112]
                logits.append(segmentor_out)
            logits = torch.cat(logits, dim=0) # shape = [40, 16, 112, 112]

        return logits
    

if __name__ == "__main__":
    model = UNETR_2D_clip(
        in_channels=1,
        out_channels=16,
        img_size=(112, 112),
        feature_size=64,
        hidden_size=768,
        mlp_dim=3072,
        num_heads=12,
        pos_embed="perceptron",
        norm_name="instance",
        conv_block=True,
        res_block=True,
        dropout_rate=0.0,
        info_mode="early",
    )

    x = torch.zeros((40, 2, 112, 112))
    logits = model(x, modality="MRI")
    print(logits.shape)