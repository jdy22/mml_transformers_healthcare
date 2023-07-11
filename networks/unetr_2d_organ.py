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
from monai.networks.nets import ViT


def add_organ_info(x, labels, organ_tokens, no_organ_tokens):
    """
    x : Feature vectors/maps to add organ information to
    labels : To indicate which organs are present, must have same batch dimension as x
    organ_tokens and no_organ_tokens : Parameter Dicts with relevant tokens to add
    Information is added along dimension 1 (patch/channel dimension)
    """
    batch_size = x.shape[0]
    x_full = x[None, 0]
    organs_present = torch.unique(labels[0])
    for organ in range(1, 16):
        if organ in organs_present:
            x_full = torch.cat((x_full, organ_tokens[str(organ)]), dim=1)
        else:
            x_full = torch.cat((x_full, no_organ_tokens[str(organ)]), dim=1)
    for i in range(1, batch_size):
        embeddings = x[None, i]
        organs_present = torch.unique(labels[i])
        for organ in range(1, 16):
            if organ in organs_present:
                embeddings = torch.cat((embeddings, organ_tokens[str(organ)]), dim=1)
            else:
                embeddings = torch.cat((embeddings, no_organ_tokens[str(organ)]), dim=1)
        x_full = torch.cat((x_full, embeddings), dim=0)
    return x_full


def add_organ_info2(x, class_logits, organ_tokens, no_organ_tokens):
    """
    x : Feature vectors/maps to add organ information to
    class_logits : To indicate which organs are present, must have same batch dimension as x
    organ_tokens and no_organ_tokens : Parameter Dicts with relevant tokens to add
    Information is added along dimension 1 (patch/channel dimension)
    """
    batch_size = x.shape[0]
    x_full = x[None, 0]
    logits = class_logits[0]
    for organ in range(1, 16):
        if logits[organ-1, 0].item() > 0:
            x_full = torch.cat((x_full, organ_tokens[str(organ)]), dim=1)
        else:
            x_full = torch.cat((x_full, no_organ_tokens[str(organ)]), dim=1)
    for i in range(1, batch_size):
        embeddings = x[None, i]
        logits = class_logits[i]
        for organ in range(1, 16):
            if logits[organ-1, 0].item() > 0:
                embeddings = torch.cat((embeddings, organ_tokens[str(organ)]), dim=1)
            else:
                embeddings = torch.cat((embeddings, no_organ_tokens[str(organ)]), dim=1)
        x_full = torch.cat((x_full, embeddings), dim=0)
    return x_full


class ViT_organ(nn.Module):
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
        classification: bool = False,
        classification_concat: bool = False
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
        
        self.hidden_size = hidden_size
        self.classification = classification
        self.classification_concat = classification_concat

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

        if self.classification:
            self.classification_tokens = nn.Parameter(torch.zeros(1, 15, hidden_size))
            self.classification_head = nn.Linear(in_features=hidden_size, out_features=1)
        elif self.classification_concat:
            self.classification_tokens = nn.Parameter(torch.zeros(1, 15, hidden_size))
            self.classification_head = nn.Linear(in_features=hidden_size, out_features=1)
            self.organ_tokens = nn.ParameterDict({
                "1" : nn.Parameter(torch.zeros(1, 1, hidden_size)),
                "2" : nn.Parameter(torch.zeros(1, 1, hidden_size)),
                "3" : nn.Parameter(torch.zeros(1, 1, hidden_size)),
                "4" : nn.Parameter(torch.zeros(1, 1, hidden_size)),
                "5" : nn.Parameter(torch.zeros(1, 1, hidden_size)),
                "6" : nn.Parameter(torch.zeros(1, 1, hidden_size)),
                "7" : nn.Parameter(torch.zeros(1, 1, hidden_size)),
                "8" : nn.Parameter(torch.zeros(1, 1, hidden_size)),
                "9" : nn.Parameter(torch.zeros(1, 1, hidden_size)),
                "10" : nn.Parameter(torch.zeros(1, 1, hidden_size)),
                "11" : nn.Parameter(torch.zeros(1, 1, hidden_size)),
                "12" : nn.Parameter(torch.zeros(1, 1, hidden_size)),
                "13" : nn.Parameter(torch.zeros(1, 1, hidden_size)),
                "14" : nn.Parameter(torch.zeros(1, 1, hidden_size)),
                "15" : nn.Parameter(torch.zeros(1, 1, hidden_size))
            })
            self.no_organ_tokens = nn.ParameterDict({
                "1" : nn.Parameter(torch.zeros(1, 1, hidden_size)),
                "2" : nn.Parameter(torch.zeros(1, 1, hidden_size)),
                "3" : nn.Parameter(torch.zeros(1, 1, hidden_size)),
                "4" : nn.Parameter(torch.zeros(1, 1, hidden_size)),
                "5" : nn.Parameter(torch.zeros(1, 1, hidden_size)),
                "6" : nn.Parameter(torch.zeros(1, 1, hidden_size)),
                "7" : nn.Parameter(torch.zeros(1, 1, hidden_size)),
                "8" : nn.Parameter(torch.zeros(1, 1, hidden_size)),
                "9" : nn.Parameter(torch.zeros(1, 1, hidden_size)),
                "10" : nn.Parameter(torch.zeros(1, 1, hidden_size)),
                "11" : nn.Parameter(torch.zeros(1, 1, hidden_size)),
                "12" : nn.Parameter(torch.zeros(1, 1, hidden_size)),
                "13" : nn.Parameter(torch.zeros(1, 1, hidden_size)),
                "14" : nn.Parameter(torch.zeros(1, 1, hidden_size)),
                "15" : nn.Parameter(torch.zeros(1, 1, hidden_size))
            })
        else:
            self.organ_tokens = nn.ParameterDict({
                "1" : nn.Parameter(torch.zeros(1, 1, hidden_size)),
                "2" : nn.Parameter(torch.zeros(1, 1, hidden_size)),
                "3" : nn.Parameter(torch.zeros(1, 1, hidden_size)),
                "4" : nn.Parameter(torch.zeros(1, 1, hidden_size)),
                "5" : nn.Parameter(torch.zeros(1, 1, hidden_size)),
                "6" : nn.Parameter(torch.zeros(1, 1, hidden_size)),
                "7" : nn.Parameter(torch.zeros(1, 1, hidden_size)),
                "8" : nn.Parameter(torch.zeros(1, 1, hidden_size)),
                "9" : nn.Parameter(torch.zeros(1, 1, hidden_size)),
                "10" : nn.Parameter(torch.zeros(1, 1, hidden_size)),
                "11" : nn.Parameter(torch.zeros(1, 1, hidden_size)),
                "12" : nn.Parameter(torch.zeros(1, 1, hidden_size)),
                "13" : nn.Parameter(torch.zeros(1, 1, hidden_size)),
                "14" : nn.Parameter(torch.zeros(1, 1, hidden_size)),
                "15" : nn.Parameter(torch.zeros(1, 1, hidden_size))
            })
            self.no_organ_tokens = nn.ParameterDict({
                "1" : nn.Parameter(torch.zeros(1, 1, hidden_size)),
                "2" : nn.Parameter(torch.zeros(1, 1, hidden_size)),
                "3" : nn.Parameter(torch.zeros(1, 1, hidden_size)),
                "4" : nn.Parameter(torch.zeros(1, 1, hidden_size)),
                "5" : nn.Parameter(torch.zeros(1, 1, hidden_size)),
                "6" : nn.Parameter(torch.zeros(1, 1, hidden_size)),
                "7" : nn.Parameter(torch.zeros(1, 1, hidden_size)),
                "8" : nn.Parameter(torch.zeros(1, 1, hidden_size)),
                "9" : nn.Parameter(torch.zeros(1, 1, hidden_size)),
                "10" : nn.Parameter(torch.zeros(1, 1, hidden_size)),
                "11" : nn.Parameter(torch.zeros(1, 1, hidden_size)),
                "12" : nn.Parameter(torch.zeros(1, 1, hidden_size)),
                "13" : nn.Parameter(torch.zeros(1, 1, hidden_size)),
                "14" : nn.Parameter(torch.zeros(1, 1, hidden_size)),
                "15" : nn.Parameter(torch.zeros(1, 1, hidden_size))
            })

    def forward(self, x_in, without_labels=False, class_layer=12):
        if self.classification or self.classification_concat or without_labels:
            x = x_in
        else:
            # First channel of x_in: image
            # Second channel of x_in: labels
            x = x_in[:, None, 0, :, :]
            labels = x_in[:, 1, :, :]
        
        x = self.patch_embedding(x)
        n_patches = x.shape[1]

        if self.classification or self.classification_concat:
            classification_tokens = self.classification_tokens.expand(x.shape[0], -1, -1)
            x_full = torch.cat((x, classification_tokens), dim=1)
        elif without_labels:
            x_full = x
        else:
            x_full = add_organ_info(x, labels, self.organ_tokens, self.no_organ_tokens)
 
        hidden_states_out = []
        class_logits = None
        for i in range(len(self.blocks)):
            x_full = self.blocks[i](x_full)
            hidden_states_out.append(x_full[:, :n_patches, :])
            if self.classification or self.classification_concat:
                if i == class_layer-1:
                    class_logits = self.classification_head(x_full[:, n_patches:, :])
                    if self.classification_concat:
                        x_full = add_organ_info2(x_full[:, :n_patches, :], class_logits, self.organ_tokens, self.no_organ_tokens)
        x_full = self.norm(x_full)
        x_out = x_full[:, :n_patches, :]

        return x_out, hidden_states_out, class_logits


class UNETR_2D_organ(nn.Module):
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
        info_mode: str = "early",
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
            info_mode: early, inter, inter2, inter3, late, classif, classif_early, classif_inter3.

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
        self.img_size = img_size
        self.feat_size = (
            img_size[0] // self.patch_size[0],
            img_size[1] // self.patch_size[1],
        )
        self.hidden_size = hidden_size
        self.info_mode = info_mode
        if self.info_mode == "classif" or self.info_mode == "classif_inter3":
            self.classification = True
            self.classification_concat = False
        elif self.info_mode == "classif_early":
            self.classification = False
            self.classification_concat = True
        else:
            self.classification = False
            self.classification_concat = False

        if self.info_mode == "early" or self.info_mode == "classif" or self.info_mode == "classif_early" or self.info_mode == "classif_inter3":
            self.vit = ViT_organ(
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
                classification=self.classification,
                classification_concat=self.classification_concat,
            )
        elif self.info_mode == "inter" or self.info_mode == "inter2" or self.info_mode == "inter3" or self.info_mode == "late":
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

        if self.info_mode == "inter" or self.info_mode == "inter2" or self.info_mode == "inter3" or self.info_mode == "classif_inter3":
            decoder_input_size_bottleneck = hidden_size + 15
            if self.info_mode == "inter" or self.info_mode == "inter2":
                decoder_input_size_skip = hidden_size + 15
            elif self.info_mode == "inter3" or self.info_mode == "classif_inter3":
                decoder_input_size_skip = hidden_size
            self.organ_tokens = nn.ParameterDict({
                "1" : nn.Parameter(torch.zeros(1, 1, self.feat_size[0], self.feat_size[1])),
                "2" : nn.Parameter(torch.zeros(1, 1, self.feat_size[0], self.feat_size[1])),
                "3" : nn.Parameter(torch.zeros(1, 1, self.feat_size[0], self.feat_size[1])),
                "4" : nn.Parameter(torch.zeros(1, 1, self.feat_size[0], self.feat_size[1])),
                "5" : nn.Parameter(torch.zeros(1, 1, self.feat_size[0], self.feat_size[1])),
                "6" : nn.Parameter(torch.zeros(1, 1, self.feat_size[0], self.feat_size[1])),
                "7" : nn.Parameter(torch.zeros(1, 1, self.feat_size[0], self.feat_size[1])),
                "8" : nn.Parameter(torch.zeros(1, 1, self.feat_size[0], self.feat_size[1])),
                "9" : nn.Parameter(torch.zeros(1, 1, self.feat_size[0], self.feat_size[1])),
                "10" : nn.Parameter(torch.zeros(1, 1, self.feat_size[0], self.feat_size[1])),
                "11" : nn.Parameter(torch.zeros(1, 1, self.feat_size[0], self.feat_size[1])),
                "12" : nn.Parameter(torch.zeros(1, 1, self.feat_size[0], self.feat_size[1])),
                "13" : nn.Parameter(torch.zeros(1, 1, self.feat_size[0], self.feat_size[1])),
                "14" : nn.Parameter(torch.zeros(1, 1, self.feat_size[0], self.feat_size[1])),
                "15" : nn.Parameter(torch.zeros(1, 1, self.feat_size[0], self.feat_size[1]))
            })
            self.no_organ_tokens = nn.ParameterDict({
                "1" : nn.Parameter(torch.zeros(1, 1, self.feat_size[0], self.feat_size[1])),
                "2" : nn.Parameter(torch.zeros(1, 1, self.feat_size[0], self.feat_size[1])),
                "3" : nn.Parameter(torch.zeros(1, 1, self.feat_size[0], self.feat_size[1])),
                "4" : nn.Parameter(torch.zeros(1, 1, self.feat_size[0], self.feat_size[1])),
                "5" : nn.Parameter(torch.zeros(1, 1, self.feat_size[0], self.feat_size[1])),
                "6" : nn.Parameter(torch.zeros(1, 1, self.feat_size[0], self.feat_size[1])),
                "7" : nn.Parameter(torch.zeros(1, 1, self.feat_size[0], self.feat_size[1])),
                "8" : nn.Parameter(torch.zeros(1, 1, self.feat_size[0], self.feat_size[1])),
                "9" : nn.Parameter(torch.zeros(1, 1, self.feat_size[0], self.feat_size[1])),
                "10" : nn.Parameter(torch.zeros(1, 1, self.feat_size[0], self.feat_size[1])),
                "11" : nn.Parameter(torch.zeros(1, 1, self.feat_size[0], self.feat_size[1])),
                "12" : nn.Parameter(torch.zeros(1, 1, self.feat_size[0], self.feat_size[1])),
                "13" : nn.Parameter(torch.zeros(1, 1, self.feat_size[0], self.feat_size[1])),
                "14" : nn.Parameter(torch.zeros(1, 1, self.feat_size[0], self.feat_size[1])),
                "15" : nn.Parameter(torch.zeros(1, 1, self.feat_size[0], self.feat_size[1]))
            })
            if self.info_mode == "inter2":
                self.organ_tokens2 = nn.ParameterDict({
                    "1" : nn.Parameter(torch.zeros(1, 1, self.feat_size[0], self.feat_size[1])),
                    "2" : nn.Parameter(torch.zeros(1, 1, self.feat_size[0], self.feat_size[1])),
                    "3" : nn.Parameter(torch.zeros(1, 1, self.feat_size[0], self.feat_size[1])),
                    "4" : nn.Parameter(torch.zeros(1, 1, self.feat_size[0], self.feat_size[1])),
                    "5" : nn.Parameter(torch.zeros(1, 1, self.feat_size[0], self.feat_size[1])),
                    "6" : nn.Parameter(torch.zeros(1, 1, self.feat_size[0], self.feat_size[1])),
                    "7" : nn.Parameter(torch.zeros(1, 1, self.feat_size[0], self.feat_size[1])),
                    "8" : nn.Parameter(torch.zeros(1, 1, self.feat_size[0], self.feat_size[1])),
                    "9" : nn.Parameter(torch.zeros(1, 1, self.feat_size[0], self.feat_size[1])),
                    "10" : nn.Parameter(torch.zeros(1, 1, self.feat_size[0], self.feat_size[1])),
                    "11" : nn.Parameter(torch.zeros(1, 1, self.feat_size[0], self.feat_size[1])),
                    "12" : nn.Parameter(torch.zeros(1, 1, self.feat_size[0], self.feat_size[1])),
                    "13" : nn.Parameter(torch.zeros(1, 1, self.feat_size[0], self.feat_size[1])),
                    "14" : nn.Parameter(torch.zeros(1, 1, self.feat_size[0], self.feat_size[1])),
                    "15" : nn.Parameter(torch.zeros(1, 1, self.feat_size[0], self.feat_size[1]))
                })
                self.no_organ_tokens2 = nn.ParameterDict({
                    "1" : nn.Parameter(torch.zeros(1, 1, self.feat_size[0], self.feat_size[1])),
                    "2" : nn.Parameter(torch.zeros(1, 1, self.feat_size[0], self.feat_size[1])),
                    "3" : nn.Parameter(torch.zeros(1, 1, self.feat_size[0], self.feat_size[1])),
                    "4" : nn.Parameter(torch.zeros(1, 1, self.feat_size[0], self.feat_size[1])),
                    "5" : nn.Parameter(torch.zeros(1, 1, self.feat_size[0], self.feat_size[1])),
                    "6" : nn.Parameter(torch.zeros(1, 1, self.feat_size[0], self.feat_size[1])),
                    "7" : nn.Parameter(torch.zeros(1, 1, self.feat_size[0], self.feat_size[1])),
                    "8" : nn.Parameter(torch.zeros(1, 1, self.feat_size[0], self.feat_size[1])),
                    "9" : nn.Parameter(torch.zeros(1, 1, self.feat_size[0], self.feat_size[1])),
                    "10" : nn.Parameter(torch.zeros(1, 1, self.feat_size[0], self.feat_size[1])),
                    "11" : nn.Parameter(torch.zeros(1, 1, self.feat_size[0], self.feat_size[1])),
                    "12" : nn.Parameter(torch.zeros(1, 1, self.feat_size[0], self.feat_size[1])),
                    "13" : nn.Parameter(torch.zeros(1, 1, self.feat_size[0], self.feat_size[1])),
                    "14" : nn.Parameter(torch.zeros(1, 1, self.feat_size[0], self.feat_size[1])),
                    "15" : nn.Parameter(torch.zeros(1, 1, self.feat_size[0], self.feat_size[1]))
                })
                self.organ_tokens3 = nn.ParameterDict({
                    "1" : nn.Parameter(torch.zeros(1, 1, self.feat_size[0], self.feat_size[1])),
                    "2" : nn.Parameter(torch.zeros(1, 1, self.feat_size[0], self.feat_size[1])),
                    "3" : nn.Parameter(torch.zeros(1, 1, self.feat_size[0], self.feat_size[1])),
                    "4" : nn.Parameter(torch.zeros(1, 1, self.feat_size[0], self.feat_size[1])),
                    "5" : nn.Parameter(torch.zeros(1, 1, self.feat_size[0], self.feat_size[1])),
                    "6" : nn.Parameter(torch.zeros(1, 1, self.feat_size[0], self.feat_size[1])),
                    "7" : nn.Parameter(torch.zeros(1, 1, self.feat_size[0], self.feat_size[1])),
                    "8" : nn.Parameter(torch.zeros(1, 1, self.feat_size[0], self.feat_size[1])),
                    "9" : nn.Parameter(torch.zeros(1, 1, self.feat_size[0], self.feat_size[1])),
                    "10" : nn.Parameter(torch.zeros(1, 1, self.feat_size[0], self.feat_size[1])),
                    "11" : nn.Parameter(torch.zeros(1, 1, self.feat_size[0], self.feat_size[1])),
                    "12" : nn.Parameter(torch.zeros(1, 1, self.feat_size[0], self.feat_size[1])),
                    "13" : nn.Parameter(torch.zeros(1, 1, self.feat_size[0], self.feat_size[1])),
                    "14" : nn.Parameter(torch.zeros(1, 1, self.feat_size[0], self.feat_size[1])),
                    "15" : nn.Parameter(torch.zeros(1, 1, self.feat_size[0], self.feat_size[1]))
                })
                self.no_organ_tokens3 = nn.ParameterDict({
                    "1" : nn.Parameter(torch.zeros(1, 1, self.feat_size[0], self.feat_size[1])),
                    "2" : nn.Parameter(torch.zeros(1, 1, self.feat_size[0], self.feat_size[1])),
                    "3" : nn.Parameter(torch.zeros(1, 1, self.feat_size[0], self.feat_size[1])),
                    "4" : nn.Parameter(torch.zeros(1, 1, self.feat_size[0], self.feat_size[1])),
                    "5" : nn.Parameter(torch.zeros(1, 1, self.feat_size[0], self.feat_size[1])),
                    "6" : nn.Parameter(torch.zeros(1, 1, self.feat_size[0], self.feat_size[1])),
                    "7" : nn.Parameter(torch.zeros(1, 1, self.feat_size[0], self.feat_size[1])),
                    "8" : nn.Parameter(torch.zeros(1, 1, self.feat_size[0], self.feat_size[1])),
                    "9" : nn.Parameter(torch.zeros(1, 1, self.feat_size[0], self.feat_size[1])),
                    "10" : nn.Parameter(torch.zeros(1, 1, self.feat_size[0], self.feat_size[1])),
                    "11" : nn.Parameter(torch.zeros(1, 1, self.feat_size[0], self.feat_size[1])),
                    "12" : nn.Parameter(torch.zeros(1, 1, self.feat_size[0], self.feat_size[1])),
                    "13" : nn.Parameter(torch.zeros(1, 1, self.feat_size[0], self.feat_size[1])),
                    "14" : nn.Parameter(torch.zeros(1, 1, self.feat_size[0], self.feat_size[1])),
                    "15" : nn.Parameter(torch.zeros(1, 1, self.feat_size[0], self.feat_size[1]))
                })
                self.organ_tokens4 = nn.ParameterDict({
                    "1" : nn.Parameter(torch.zeros(1, 1, self.feat_size[0], self.feat_size[1])),
                    "2" : nn.Parameter(torch.zeros(1, 1, self.feat_size[0], self.feat_size[1])),
                    "3" : nn.Parameter(torch.zeros(1, 1, self.feat_size[0], self.feat_size[1])),
                    "4" : nn.Parameter(torch.zeros(1, 1, self.feat_size[0], self.feat_size[1])),
                    "5" : nn.Parameter(torch.zeros(1, 1, self.feat_size[0], self.feat_size[1])),
                    "6" : nn.Parameter(torch.zeros(1, 1, self.feat_size[0], self.feat_size[1])),
                    "7" : nn.Parameter(torch.zeros(1, 1, self.feat_size[0], self.feat_size[1])),
                    "8" : nn.Parameter(torch.zeros(1, 1, self.feat_size[0], self.feat_size[1])),
                    "9" : nn.Parameter(torch.zeros(1, 1, self.feat_size[0], self.feat_size[1])),
                    "10" : nn.Parameter(torch.zeros(1, 1, self.feat_size[0], self.feat_size[1])),
                    "11" : nn.Parameter(torch.zeros(1, 1, self.feat_size[0], self.feat_size[1])),
                    "12" : nn.Parameter(torch.zeros(1, 1, self.feat_size[0], self.feat_size[1])),
                    "13" : nn.Parameter(torch.zeros(1, 1, self.feat_size[0], self.feat_size[1])),
                    "14" : nn.Parameter(torch.zeros(1, 1, self.feat_size[0], self.feat_size[1])),
                    "15" : nn.Parameter(torch.zeros(1, 1, self.feat_size[0], self.feat_size[1]))
                })
                self.no_organ_tokens4 = nn.ParameterDict({
                    "1" : nn.Parameter(torch.zeros(1, 1, self.feat_size[0], self.feat_size[1])),
                    "2" : nn.Parameter(torch.zeros(1, 1, self.feat_size[0], self.feat_size[1])),
                    "3" : nn.Parameter(torch.zeros(1, 1, self.feat_size[0], self.feat_size[1])),
                    "4" : nn.Parameter(torch.zeros(1, 1, self.feat_size[0], self.feat_size[1])),
                    "5" : nn.Parameter(torch.zeros(1, 1, self.feat_size[0], self.feat_size[1])),
                    "6" : nn.Parameter(torch.zeros(1, 1, self.feat_size[0], self.feat_size[1])),
                    "7" : nn.Parameter(torch.zeros(1, 1, self.feat_size[0], self.feat_size[1])),
                    "8" : nn.Parameter(torch.zeros(1, 1, self.feat_size[0], self.feat_size[1])),
                    "9" : nn.Parameter(torch.zeros(1, 1, self.feat_size[0], self.feat_size[1])),
                    "10" : nn.Parameter(torch.zeros(1, 1, self.feat_size[0], self.feat_size[1])),
                    "11" : nn.Parameter(torch.zeros(1, 1, self.feat_size[0], self.feat_size[1])),
                    "12" : nn.Parameter(torch.zeros(1, 1, self.feat_size[0], self.feat_size[1])),
                    "13" : nn.Parameter(torch.zeros(1, 1, self.feat_size[0], self.feat_size[1])),
                    "14" : nn.Parameter(torch.zeros(1, 1, self.feat_size[0], self.feat_size[1])),
                    "15" : nn.Parameter(torch.zeros(1, 1, self.feat_size[0], self.feat_size[1]))
                })
        elif self.info_mode == "early" or self.info_mode == "late" or self.info_mode == "classif" or self.info_mode == "classif_early":
            decoder_input_size_bottleneck = hidden_size
            decoder_input_size_skip = hidden_size
        self.encoder2 = UnetrPrUpBlock(
            spatial_dims=2,
            in_channels=decoder_input_size_skip,
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
            in_channels=decoder_input_size_skip,
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
            in_channels=decoder_input_size_skip,
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
            in_channels=decoder_input_size_bottleneck,
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

        if self.info_mode == "late":
            output_input_size = feature_size + 15
            self.organ_tokens = nn.ParameterDict({
                "1" : nn.Parameter(torch.zeros(1, 1, self.img_size[0], self.img_size[1])),
                "2" : nn.Parameter(torch.zeros(1, 1, self.img_size[0], self.img_size[1])),
                "3" : nn.Parameter(torch.zeros(1, 1, self.img_size[0], self.img_size[1])),
                "4" : nn.Parameter(torch.zeros(1, 1, self.img_size[0], self.img_size[1])),
                "5" : nn.Parameter(torch.zeros(1, 1, self.img_size[0], self.img_size[1])),
                "6" : nn.Parameter(torch.zeros(1, 1, self.img_size[0], self.img_size[1])),
                "7" : nn.Parameter(torch.zeros(1, 1, self.img_size[0], self.img_size[1])),
                "8" : nn.Parameter(torch.zeros(1, 1, self.img_size[0], self.img_size[1])),
                "9" : nn.Parameter(torch.zeros(1, 1, self.img_size[0], self.img_size[1])),
                "10" : nn.Parameter(torch.zeros(1, 1, self.img_size[0], self.img_size[1])),
                "11" : nn.Parameter(torch.zeros(1, 1, self.img_size[0], self.img_size[1])),
                "12" : nn.Parameter(torch.zeros(1, 1, self.img_size[0], self.img_size[1])),
                "13" : nn.Parameter(torch.zeros(1, 1, self.img_size[0], self.img_size[1])),
                "14" : nn.Parameter(torch.zeros(1, 1, self.img_size[0], self.img_size[1])),
                "15" : nn.Parameter(torch.zeros(1, 1, self.img_size[0], self.img_size[1]))
            })
            self.no_organ_tokens = nn.ParameterDict({
                "1" : nn.Parameter(torch.zeros(1, 1, self.img_size[0], self.img_size[1])),
                "2" : nn.Parameter(torch.zeros(1, 1, self.img_size[0], self.img_size[1])),
                "3" : nn.Parameter(torch.zeros(1, 1, self.img_size[0], self.img_size[1])),
                "4" : nn.Parameter(torch.zeros(1, 1, self.img_size[0], self.img_size[1])),
                "5" : nn.Parameter(torch.zeros(1, 1, self.img_size[0], self.img_size[1])),
                "6" : nn.Parameter(torch.zeros(1, 1, self.img_size[0], self.img_size[1])),
                "7" : nn.Parameter(torch.zeros(1, 1, self.img_size[0], self.img_size[1])),
                "8" : nn.Parameter(torch.zeros(1, 1, self.img_size[0], self.img_size[1])),
                "9" : nn.Parameter(torch.zeros(1, 1, self.img_size[0], self.img_size[1])),
                "10" : nn.Parameter(torch.zeros(1, 1, self.img_size[0], self.img_size[1])),
                "11" : nn.Parameter(torch.zeros(1, 1, self.img_size[0], self.img_size[1])),
                "12" : nn.Parameter(torch.zeros(1, 1, self.img_size[0], self.img_size[1])),
                "13" : nn.Parameter(torch.zeros(1, 1, self.img_size[0], self.img_size[1])),
                "14" : nn.Parameter(torch.zeros(1, 1, self.img_size[0], self.img_size[1])),
                "15" : nn.Parameter(torch.zeros(1, 1, self.img_size[0], self.img_size[1]))
            })
        elif self.info_mode == "early" or self.info_mode == "inter" or self.info_mode == "inter2" or self.info_mode == "inter3" or self.info_mode == "classif" or self.info_mode == "classif_early" or self.info_mode == "classif_inter3":
            output_input_size = feature_size
        self.out = UnetOutBlock(spatial_dims=2, in_channels=output_input_size, out_channels=out_channels)

    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], hidden_size)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

    def forward(self, x_in, without_labels=False, test_mode=False, class_layer=12):
        if self.info_mode == "early" or self.info_mode == "classif" or self.info_mode == "classif_early" or self.info_mode == "classif_inter3":
            x, hidden_states_out, class_logits = self.vit(x_in, without_labels, class_layer)
        elif self.info_mode == "inter" or self.info_mode == "inter2" or self.info_mode == "inter3" or self.info_mode == "late":
            x = x_in[:, None, 0, :, :]
            labels = x_in[:, 1, :, :]
            x, hidden_states_out = self.vit(x)

        if self.classification or self.classification_concat or without_labels:
            enc1 = self.encoder1(x_in)
        else:
            enc1 = self.encoder1(x_in[:, None, 0, :, :])

        x2 = hidden_states_out[2]
        x2 = self.proj_feat(x2, self.hidden_size, self.feat_size)
        if self.info_mode == "inter":
            x2 = add_organ_info(x2, labels, self.organ_tokens, self.no_organ_tokens)
        elif self.info_mode == "inter2":
            x2 = add_organ_info(x2, labels, self.organ_tokens2, self.no_organ_tokens2)
        enc2 = self.encoder2(x2)

        x3 = hidden_states_out[5]
        x3 = self.proj_feat(x3, self.hidden_size, self.feat_size)
        if self.info_mode == "inter":
            x3 = add_organ_info(x3, labels, self.organ_tokens, self.no_organ_tokens)
        elif self.info_mode == "inter2":
            x3 = add_organ_info(x3, labels, self.organ_tokens3, self.no_organ_tokens3)
        enc3 = self.encoder3(x3)

        x4 = hidden_states_out[8]
        x4 = self.proj_feat(x4, self.hidden_size, self.feat_size)
        if self.info_mode == "inter":
            x4 = add_organ_info(x4, labels, self.organ_tokens, self.no_organ_tokens)
        elif self.info_mode == "inter2":
            x4 = add_organ_info(x4, labels, self.organ_tokens4, self.no_organ_tokens4)
        enc4 = self.encoder4(x4)

        dec4 = self.proj_feat(x, self.hidden_size, self.feat_size)
        if self.info_mode == "inter" or self.info_mode == "inter2" or self.info_mode == "inter3":
            dec4 = add_organ_info(dec4, labels, self.organ_tokens, self.no_organ_tokens)
        elif self.info_mode == "classif_inter3":
            dec4 = add_organ_info2(dec4, class_logits, self.organ_tokens, self.no_organ_tokens)
        dec3 = self.decoder5(dec4, enc4)

        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        out = self.decoder2(dec1, enc1)

        if self.info_mode == "late":
            out = add_organ_info(out, labels, self.organ_tokens, self.no_organ_tokens)
        seg_logits = self.out(out)

        if self.classification or self.classification_concat:
            if test_mode:
                return seg_logits
            else:
                return seg_logits, class_logits
        else:
            return seg_logits
    

if __name__ == "__main__":
    model = UNETR_2D_organ(
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
        info_mode="classif_inter3",
    )

    # x = torch.zeros((40, 2, 112, 112))
    # logits = model(x)
    # print(logits.shape)

    # x = torch.zeros((40, 1, 112, 112))
    # logits = model(x, without_labels=True)
    # print(logits.shape)

    x = torch.zeros((40, 1, 112, 112))
    seg_logits, class_logits = model(x, test_mode=False, class_layer=12)
    print(seg_logits.shape)
    print(class_logits.shape)