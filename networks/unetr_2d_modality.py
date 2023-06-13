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

from typing import Tuple, Union

from collections.abc import Sequence

import torch
import torch.nn as nn

from monai.networks.blocks.patchembedding import PatchEmbeddingBlock
from monai.networks.blocks.transformerblock import TransformerBlock
from monai.networks.blocks import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock
from monai.networks.blocks.dynunet_block import UnetOutBlock


class ViT_modality(nn.Module):
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

        self.CT_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.MRI_token = nn.Parameter(torch.zeros(1, 1, hidden_size))

    def forward(self, x, modality, info_mode):
        # Options for modality: "CT" or "MRI"
        # Options for info_mode: "concat" or "add"
        x = self.patch_embedding(x)
        if info_mode == "concat":
            if modality == "CT":
                modality_token = self.CT_token.expand(x.shape[0], -1, -1)
            elif modality == "MRI":
                modality_token = self.MRI_token.expand(x.shape[0], -1, -1)
            x_full = torch.cat((x, modality_token), dim=1)
        elif info_mode == "add":
            if modality == "CT":
                modality_token = self.CT_token.expand(x.shape[0], x.shape[1], -1)
            elif modality == "MRI":
                modality_token = self.MRI_token.expand(x.shape[0], x.shape[1], -1)
            x_full = x + modality_token
 
        hidden_states_out = []
        for blk in self.blocks:
            x_full = blk(x_full)
            if info_mode == "concat":
                hidden_states_out.append(x_full[:, :-1, :])
            elif info_mode == "add":
                hidden_states_out.append(x_full)
        x_full = self.norm(x_full)
        if info_mode == "concat":
            x_out = x_full[:, :-1, :]
            modality_out = x_full[:, -1, :]
        elif info_mode == "add":
            x_out = x_full
            modality_out = None

        return x_out, hidden_states_out, modality_out


class UNETR_2D_modality(nn.Module):
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
        self.vit = ViT_modality(
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
        self.out = UnetOutBlock(spatial_dims=2, in_channels=feature_size, out_channels=out_channels)  # type: ignore

    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], hidden_size)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

    def forward(self, x_in, modality, info_mode):
        x, hidden_states_out, modality_out = self.vit(x_in, modality, info_mode)
        enc1 = self.encoder1(x_in)
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
        out = self.decoder2(dec1, enc1)
        logits = self.out(out)
        return logits
    

if __name__ == "__main__":
    model = UNETR_2D_modality(
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
    )

    x = torch.zeros((40, 1, 112, 112))
    logits = model(x, modality="CT", info_mode="add")
    print(logits.shape)