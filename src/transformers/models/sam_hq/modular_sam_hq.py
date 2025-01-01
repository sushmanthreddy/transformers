# coding=utf-8
# Copyright 2024 Google Inc. HuggingFace Inc. team. All rights reserved.
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass
from typing import Optional, List , Tuple,Dict, Union
import torch
import numpy as np
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn,Tensor

from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from ..sam.modeling_sam import SamVisionEncoderOutput, SamImageSegmentationOutput, SamPatchEmbeddings, SamMLPBlock,SamVisionAttention,SamVisionLayer,SamLayerNorm,SamVisionNeck,SamMaskEmbedding,SamPromptEncoder,SamTwoWayTransformer,SamPositionalEmbedding,SamFeedForward,SamPreTrainedModel,SAM_START_DOCSTRING,SAM_INPUTS_DOCSTRING
from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ..sam.configuration_sam import SamPromptEncoderConfig, SamVisionConfig, SamConfig




logger = logging.get_logger(__name__)



class SamHQPromptEncoderConfig(SamPromptEncoderConfig):
    r"""
    This is the configuration class to store the configuration of a [`SamHQPromptEncoderModel`].The [`SamHQPromptEncoderModel`]
    module is used to encode the input 2D points and bounding boxes. Instantiating a configuration defaults will yield a
    similar configuration to that of the SAM_HQ model. The configuration is used to store the configuration of the model.
    [Uminosachi/sam-hq](https://huggingface.co/Uminosachi/sam-hq) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model's output.Read the documentation from
    [`PretrainedConfig`] for more information.

    Args:
        hidden_size (`int`, *optional*, defaults to 256):
            Dimensionality of the hidden states.
        image_size (`int`, *optional*, defaults to 1024):
            The expected output resolution of the image.
        patch_size (`int`, *optional*, defaults to 16):
            The size (resolution) of each patch.
        mask_input_channels (`int`, *optional*, defaults to 16):
            The number of channels to be fed to the `MaskDecoder` module.
        num_point_embeddings (`int`, *optional*, defaults to 4):
            The number of point embeddings to be used.
        hidden_act (`str`, *optional*, defaults to `"gelu"`):
            The non-linear activation function in the encoder and pooler.
    """
    pass
    



class SamHQVisionConfig(SamVisionConfig):
    r"""
    This is the configuration class to store the configuration of a [`SamHQVisionModel`]. It is used to instantiate a
    SAM HQ vision encoder according to the specified arguments, defining the model architecture. Instantiating a
    configuration with defaults will yield a configuration similar to that of the SAM HQ Vision model.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        output_channels (`int`, *optional*, defaults to 256):
            Dimensionality of the output channels in the Patch Encoder.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_channels (`int`, *optional*, defaults to 3):
            Number of channels in the input image.
        image_size (`int`, *optional*, defaults to 1024):
            Expected resolution. Target size of the resized input image.
        patch_size (`int`, *optional*, defaults to 16):
            Size of the patches to be extracted from the input image.
        hidden_act (`str`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string).
        layer_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the layer normalization layers.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 1e-10):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        qkv_bias (`bool`, *optional*, defaults to `True`):
            Whether to add a bias to query, key, value projections.
        mlp_ratio (`float`, *optional*, defaults to 4.0):
            Ratio of mlp hidden dim to embedding dim.
        use_abs_pos (`bool`, *optional*, defaults to `True`):
            Whether to use absolute position embedding.
        use_rel_pos (`bool`, *optional*, defaults to `True`):
            Whether to use relative position embedding.
        window_size (`int`, *optional*, defaults to 14):
            Window size for relative position.
        global_attn_indexes (`List[int]`, *optional*, defaults to `[2, 5, 8, 11]`):
            The indexes of the global attention layers.
        num_pos_feats (`int`, *optional*, defaults to 128):
            The dimensionality of the position embedding.
        mlp_dim (`int`, *optional*):
            The dimensionality of the MLP layer in the Transformer encoder. If `None`, defaults to `mlp_ratio *
            hidden_size`.
    """
    pass





class SamHQMaskDecoderConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`SamHQMaskDecoder`]. It is used to instantiate a
    SAM HQ mask decoder with the specified arguments, defining the model architecture. Instantiating a configuration
    with defaults will yield a configuration similar to that of the SAM HQ variant.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        hidden_size (`int`, *optional*, defaults to 256):
            Dimensionality of the hidden states.
        hidden_act (`str`, *optional*, defaults to `"relu"`):
            The non-linear activation function used inside the `SamHQMaskDecoder` module.
        mlp_dim (`int`, *optional*, defaults to 2048):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        num_hidden_layers (`int`, *optional*, defaults to 2):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads for each attention layer in the Transformer encoder.
        attention_downsample_rate (`int`, *optional*, defaults to 2):
            The downsampling rate of the attention layer.
        num_multimask_outputs (`int`, *optional*, defaults to 3):
            The number of outputs from the `SamHQMaskDecoder` module. In the Segment Anything paper, this is set to 3.
        iou_head_depth (`int`, *optional*, defaults to 3):
            The number of layers in the IoU head module.
        iou_head_hidden_dim (`int`, *optional*, defaults to 256):
            The dimensionality of the hidden states in the IoU head module.
        layer_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the layer normalization layers.
        vit_dim (`int`, *optional*, defaults to 768):
            Dimensionality of the Vision Transformer (ViT) used in the `SamHQMaskDecoder` module.
    """

    def __init__(
        self,
        hidden_size=256,
        hidden_act="relu",
        mlp_dim=2048,
        num_hidden_layers=2,
        num_attention_heads=8,
        attention_downsample_rate=2,
        num_multimask_outputs=3,
        iou_head_depth=3,
        iou_head_hidden_dim=256,
        layer_norm_eps=1e-6,
        vit_dim=768,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.mlp_dim = mlp_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.attention_downsample_rate = attention_downsample_rate
        self.num_multimask_outputs = num_multimask_outputs
        self.iou_head_depth = iou_head_depth
        self.iou_head_hidden_dim = iou_head_hidden_dim
        self.layer_norm_eps = layer_norm_eps
        self.vit_dim = vit_dim


class SamHQConfig(SamConfig):
    r"""
    [`SamHQConfig`] is the configuration class to store the configuration of a [`SamHQModel`]. It is used to instantiate a
    SAM-HQ model according to the specified arguments, defining the vision model, prompt-encoder model and mask decoder
    configs. Instantiating a configuration with the defaults will yield a similar configuration to that of the
    SAM-HQ-ViT-H [sushmanth/sam_hq_vit_h](https://huggingface.co/sushmanth/sam_hq_vit_h) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vision_config (Union[`dict`, `SamHQVisionConfig`], *optional*):
            Dictionary of configuration options used to initialize [`SamHQVisionConfig`].
        prompt_encoder_config (Union[`dict`, `SamHQPromptEncoderConfig`], *optional*):
            Dictionary of configuration options used to initialize [`SamHQPromptEncoderConfig`].
        mask_decoder_config (Union[`dict`, `SamHQMaskDecoderConfig`], *optional*):
            Dictionary of configuration options used to initialize [`SamHQMaskDecoderConfig`].
        kwargs (*optional*):
            Dictionary of keyword arguments.
    """

    model_type = "sam-hq"

    def __init__(
        self,
        vision_config=None,
        prompt_encoder_config=None,
        mask_decoder_config=None,
        initializer_range=0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)
        vision_config = vision_config if vision_config is not None else {}
        prompt_encoder_config = prompt_encoder_config if prompt_encoder_config is not None else {}
        mask_decoder_config = mask_decoder_config if mask_decoder_config is not None else {}

        if isinstance(vision_config, SamHQVisionConfig):
            vision_config = vision_config.to_dict()
        if isinstance(prompt_encoder_config, SamHQPromptEncoderConfig):
            prompt_encoder_config = prompt_encoder_config.to_dict()
        if isinstance(mask_decoder_config, SamHQMaskDecoderConfig):
            mask_decoder_config = mask_decoder_config.to_dict()

        self.vision_config = SamHQVisionConfig(**vision_config)
        self.prompt_encoder_config = SamHQPromptEncoderConfig(**prompt_encoder_config)
        self.mask_decoder_config = SamHQMaskDecoderConfig(**mask_decoder_config)
        self.initializer_range = initializer_range



@dataclass
class SamHQVisionEncoderOutput(ModelOutput):
    """
    Base class for SAM-HQ vision model's outputs. Inherits from SamVisionEncoderOutput with additional field for intermediate embeddings.

    Args:
        image_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim)`, *optional*):
            The image embeddings obtained by applying the projection layer to the pooler_output.
        
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
            
        intermediate_embeddings (`list(torch.FloatTensor)`, *optional*):
            A list of intermediate embeddings collected from certain blocks within the model, typically those without
            windowed attention. Each element in the list is of shape `(batch_size, sequence_length, hidden_size)`.
            This is specific to SAM-HQ and not present in base SAM.
            
        hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

        attentions (`tuple(torch.FloatTensor)`, *optional*):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
    """
    image_embeds: Optional[torch.FloatTensor] = None
    intermediate_embeddings: Optional[List[torch.FloatTensor]] = None
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


@dataclass
class SamHQImageSegmentationOutput(SamImageSegmentationOutput):
    """
    Base class for Segment-Anything High Quality model's output

    Args:
        iou_scores (`torch.FloatTensor` of shape `(batch_size, num_masks)`):
            The iou scores of the predicted masks.
        pred_masks (`torch.FloatTensor` of shape `(batch_size, num_masks, height, width)`):
            The predicted low resolutions masks. Needs to be post-processed by the processor
        vision_hidden_states  (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the vision model at the output of each layer plus the optional initial embedding outputs.
        vision_attentions  (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        mask_decoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """
    pass

class SamHQPatchEmbeddings(SamPatchEmbeddings):
    pass



class SamHQMLPBlock(SamMLPBlock):
    pass


class SamHQVisionAttention(SamVisionAttention):
    pass

class SamHQVisionLayer(SamVisionLayer):
    pass


class SamHQLayerNorm(SamLayerNorm):
    pass

class SamHQVisionNeck(SamVisionNeck):
    pass
 

class SamHQVisionEncoder(nn.Module):
    def __init__(self, config: SamHQVisionConfig):
        super().__init__()
        self.config = config
        self.image_size = config.image_size

        self.patch_embed = SamHQPatchEmbeddings(config)

        self.pos_embed = None

        if config.use_abs_pos:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, 
                            config.image_size // config.patch_size,
                            config.image_size // config.patch_size,
                            config.hidden_size,
                )
            )


        self.layers = nn.ModuleList()
        for i in range(config.num_hidden_layers):
            layer = SamHQVisionLayer(
                config,
                window_size=config.window_size if i not in config.global_attn_indexes else 0,
            )
            self.layers.append(layer)

        self.neck = SamHQVisionNeck(config)

        self.gradient_checkpointing = False


    def get_input_embeddings(self):
        return self.patch_embed
    

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SamHQVisionEncoderOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        hidden_states = self.patch_embed(pixel_values)
        if self.pos_embed is not None:
            hidden_states = hidden_states + self.pos_embed

        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        intermediate_embeddings = []

        for i, layer_module in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                )
            else:
                layer_outputs = layer_module(hidden_states, output_attentions= output_attentions)

            hidden_states = layer_outputs[0]

            # Collect embeddings from non-windowed blocks
            if hasattr(layer_module, 'window_size') and layer_module.window_size == 0:
                intermediate_embeddings.append(hidden_states)

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)


        hidden_states = self.neck(hidden_states)

        image_embeddings = hidden_states

        if not return_dict:
            outputs = (image_embeddings, intermediate_embeddings)
            if output_hidden_states:
                outputs = outputs + (all_hidden_states,)
            if output_attentions:
                outputs = outputs + (all_self_attentions,)
            return outputs

        return SamHQVisionEncoderOutput(
            image_embeds = image_embeddings,
            last_hidden_state=hidden_states,
            intermediate_embeddings=intermediate_embeddings,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
    

class SamHQMaskEmbedding(SamMaskEmbedding):
    pass


class SamHQPromptEncoder(SamPromptEncoder):
    pass


class SamHQAttention(nn.Module):
    """
    SAM_HQ's attention layer that allows for downscaling the size of the embedding after projection to queries, keys, and
    values.
    """

    def __init__(self, config, downsample_rate=None):
        super().__init__()
        self.hidden_size = config.hidden_size

        downsample_rate = config.attention_downsample_rate if downsample_rate is None else downsample_rate

        self.internal_dim = config.hidden_size // downsample_rate
        self.num_attention_heads = config.num_attention_heads
        if self.internal_dim % config.num_attention_heads != 0:
            raise ValueError("num_attention_heads must divide hidden_size.")

        self.q_proj = nn.Linear(self.hidden_size, self.internal_dim)
        self.k_proj = nn.Linear(self.hidden_size, self.internal_dim)
        self.v_proj = nn.Linear(self.hidden_size, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, self.hidden_size)

    def _separate_heads(self, hidden_states: Tensor, num_attention_heads: int) -> Tensor:
        batch, point_batch_size, n_tokens, channel = hidden_states.shape
        c_per_head = channel // num_attention_heads
        hidden_states = hidden_states.reshape(batch * point_batch_size, n_tokens, num_attention_heads, c_per_head)
        return hidden_states.transpose(1, 2)

    def _recombine_heads(self, hidden_states: Tensor, point_batch_size: int) -> Tensor:
        batch, n_heads, n_tokens, c_per_head = hidden_states.shape
        hidden_states = hidden_states.transpose(1, 2)
        return hidden_states.reshape(batch // point_batch_size, point_batch_size, n_tokens, n_heads * c_per_head)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, attention_similarity: Tensor = None) -> Tensor:
        # Input projections
        query = self.q_proj(query)
        key = self.k_proj(key)
        value = self.v_proj(value)

        point_batch_size = query.shape[1]
        # Separate into heads
        query = self._separate_heads(query, self.num_attention_heads)
        key = self._separate_heads(key, self.num_attention_heads)
        value = self._separate_heads(value, self.num_attention_heads)

        # SamAttention
        _, _, _, c_per_head = query.shape
        attn = query @ key.permute(0, 1, 3, 2)  # batch_size * point_batch_size  x N_heads x N_tokens x N_tokens
        attn = attn / (c_per_head**0.5)
        attn = torch.softmax(attn, dim=-1)

        if attention_similarity is not None:
            attn = attn + attention_similarity
            attn = torch.softmax(attn, dim=-1)

        # Get output
        out = attn @ value
        out = self._recombine_heads(out, point_batch_size)
        out = self.out_proj(out)

        return out    


class SamHQTwoWayAttentionBlock(nn.Module):
    def __init__(self, config, attention_downsample_rate: int = 2, skip_first_layer_pe: bool = False):
        """
        A transformer block with four layers:
            (1) self-attention of sparse inputs (2) cross attention of sparse inputs -> dense inputs (3) mlp block on
            sparse inputs (4) cross attention of dense inputs -> sparse inputs

        Arguments:
            config (`SamHQMaskDecoderConfig`):
                The configuration file used to instantiate the block
            attention_downsample_rate (*optionalk*, int, defaults to 2):
                The downsample ratio of the block used to reduce the inner dim of the attention.
            skip_first_layer_pe (*optional*, bool, defaults to `False`):
                Whether or not to skip the addition of the query_point_embedding on the first layer.
        """
        super().__init__()

        self.hidden_size = config.hidden_size
        self.layer_norm_eps = config.layer_norm_eps

        self.self_attn = SamHQAttention(config, downsample_rate=1)
        self.layer_norm1 = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)

        self.cross_attn_token_to_image = SamHQAttention(config, downsample_rate=attention_downsample_rate)
        self.layer_norm2 = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)

        self.mlp = SamHQMLPBlock(config)
        self.layer_norm3 = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)

        self.layer_norm4 = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.cross_attn_image_to_token = SamHQAttention(config, downsample_rate=attention_downsample_rate)

        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(
        self,
        queries: Tensor,
        keys: Tensor,
        query_point_embedding: Tensor,
        key_point_embedding: Tensor,
        attention_similarity: Tensor,
        output_attentions: bool = False,
    ):
        # Self attention block
        if self.skip_first_layer_pe:
            queries = self.self_attn(query=queries, key=queries, value=queries)
        else:
            query = queries + query_point_embedding
            attn_out = self.self_attn(query=query, key=query, value=queries)
            queries = queries + attn_out
        queries = self.layer_norm1(queries)

        # Cross attention block, tokens attending to image embedding
        query = queries + query_point_embedding
        key = keys + key_point_embedding

        attn_out = self.cross_attn_token_to_image(
            query=query, key=key, value=keys, attention_similarity=attention_similarity
        )
        queries = queries + attn_out

        queries = self.layer_norm2(queries)

        # MLP block
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.layer_norm3(queries)

        # Cross attention block, image embedding attending to tokens
        query = queries + query_point_embedding
        key = keys + key_point_embedding

        attn_out = self.cross_attn_image_to_token(query=key, key=query, value=queries)
        keys = keys + attn_out

        keys = self.layer_norm4(keys)

        outputs = (queries, keys)

        if output_attentions:
            outputs = outputs + (attn_out,)
        else:
            outputs = outputs + (None,)

        return outputs

class SamHQPositionalEmbedding(SamPositionalEmbedding):
    pass

class SamHQTwoWayTransformer(SamTwoWayTransformer):
    pass


class SamHQFeedForward(SamFeedForward):
    pass


class SamHQMaskDecoder(nn.Module):
    def __init__(self, config: SamHQMaskDecoderConfig):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.num_multimask_outputs = config.num_multimask_outputs
        self.num_mask_tokens = config.num_multimask_outputs + 1

        self.iou_token = nn.Embedding(1, self.hidden_size)
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, self.hidden_size)


        self.transformer = SamHQTwoWayTransformer(config)

        self.upscale_conv1 = nn.ConvTranspose2d(self.hidden_size, self.hidden_size // 4, kernel_size=2, stride=2)
        self.upscale_conv2 = nn.ConvTranspose2d(self.hidden_size // 4, self.hidden_size // 8, kernel_size=2, stride=2)
        self.upscale_layer_norm = SamHQLayerNorm(self.hidden_size // 4, data_format="channels_first")
        self.activation = nn.GELU()


        mlps_list = []
        for _ in range(self.num_mask_tokens):
            mlps_list += [SamHQFeedForward(self.hidden_size, self.hidden_size, self.hidden_size // 8, 3)]
        self.output_hypernetworks_mlps = nn.ModuleList(mlps_list)

        self.iou_prediction_head = SamHQFeedForward(
            self.hidden_size, config.iou_head_hidden_dim, self.num_mask_tokens, config.iou_head_depth
        )


        self.hq_token = nn.Embedding(1, self.hidden_size)
        self.hf_mlp = SamHQFeedForward(self.hidden_size, self.hidden_size, self.hidden_size // 8, 3)
        self.num_mask_tokens = self.num_mask_tokens + 1

        # Compress ViT features
        self.compress_vit_conv1 = nn.ConvTranspose2d(config.vit_dim, self.hidden_size, kernel_size=2, stride=2)
        self.compress_vit_norm = SamHQLayerNorm(self.hidden_size, data_format="channels_first")
        self.compress_vit_conv2 = nn.ConvTranspose2d(self.hidden_size, self.hidden_size // 8, kernel_size=2, stride=2)
        self.activation = nn.GELU()

        # Embedding encoder
        self.encoder_conv1 = nn.ConvTranspose2d(self.hidden_size, self.hidden_size // 4, kernel_size=2, stride=2)
        self.encoder_norm = SamHQLayerNorm(self.hidden_size // 4, data_format="channels_first")
        self.encoder_conv2 = nn.ConvTranspose2d(self.hidden_size // 4, self.hidden_size // 8, kernel_size=2, stride=2)

        # Embedding mask feature
        self.mask_conv1 = nn.Conv2d(self.hidden_size // 8, self.hidden_size // 4, kernel_size=3, stride=1, padding=1)
        self.mask_norm = SamHQLayerNorm(self.hidden_size // 4, data_format="channels_first")
        self.mask_conv2 = nn.Conv2d(self.hidden_size // 4, self.hidden_size // 8, kernel_size=3, stride=1, padding=1)


    def forward(
            self,
            image_embeddings: torch.Tensor,
            image_positional_embeddings: torch.Tensor,
            sparse_prompt_embeddings: torch.Tensor,
            dense_prompt_embeddings: torch.Tensor,
            multimask_output: bool,
            hq_token_only: bool,
            interm_embeddings: torch.Tensor,
            output_attentions: Optional[bool] = None,
            attention_similarity: torch.Tensor = None,
            target_embedding: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
    Predict high-quality masks given image and prompt embeddings.

    Args:
        image_embeddings (`torch.Tensor`):
            The embeddings from the image encoder.
        image_positional_embedding (`torch.Tensor`):
            Positional encoding with the shape of image_embeddings.
        sparse_prompt_embeddings (`torch.Tensor`):
            The embeddings of the points and boxes.
        dense_prompt_embeddings (`torch.Tensor`):
            The embeddings of the mask inputs.
        multimask_output (bool):
            Whether to return multiple masks or a single mask.
        hq_token_only (bool): 
            Whether to use only the high-quality token output or combine with SAM output.
        interm_embeddings (`torch.Tensor`):
            Intermediate embeddings from the vision encoder for feature fusion.
        output_attentions (bool, *optional*):
            Whether or not to return the attentions tensors of all attention layers.
        attention_similarity (`torch.Tensor`, *optional*):
            Optional tensor for attention similarity computation.
        target_embedding (`torch.Tensor`, *optional*):
            Optional target embedding for transformer processing.
            
    Returns:
        `Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]`: A tuple of tensors containing:
            - A tensor of shape `(batch_size, num_prompts, num_masks, height, width)` containing the output masks.
            - A tensor of shape `(batch_size, num_prompts, num_masks)` containing the iou predictions for each mask.
            - (Optional) A tuple containing attention tensors if output_attentions is True.
    """
        batch_size, num_channels, height, width = image_embeddings.shape
        point_batch_size = sparse_prompt_embeddings.shape[1]
        vit_features = interm_embeddings[0].permute(0, 3, 1, 2)

        embed_encode = self.encoder_conv1(image_embeddings)
        embed_encode = self.activation(self.encoder_norm(embed_encode))
        embed_encode = self.encoder_conv2(embed_encode)

        com_vit_feat = self.compress_vit_conv1(vit_features)
        com_vit_feat = self.activation(self.compress_vit_norm(com_vit_feat))
        com_vit_feat = self.compress_vit_conv2(com_vit_feat)

        hq_features = embed_encode + com_vit_feat

        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight, self.hq_token.weight], dim=0)
        output_tokens = output_tokens.repeat(batch_size, point_batch_size, 1,  1)

        if sparse_prompt_embeddings.sum().item() != 0:
            tokens = torch.cat([output_tokens, sparse_prompt_embeddings], dim=2)
        else:
            tokens = output_tokens
        point_embeddings = tokens.to(self.iou_token.weight.dtype)


        image_embeddings = image_embeddings + dense_prompt_embeddings
        image_embeddings = image_embeddings.repeat_interleave(point_batch_size, 0)
        image_positional_embeddings = image_positional_embeddings.repeat_interleave(point_batch_size, 0)

        point_embedding, image_embeddings, attentions = self.transformer(
            point_embeddings=point_embeddings,
            image_embeddings=image_embeddings,
            image_positional_embeddings=image_positional_embeddings,
            attention_similarity=attention_similarity,
            target_embedding=target_embedding,
            output_attentions=output_attentions,
        )
        iou_token_out = point_embedding[:, :, 0, :]
        mask_tokens_out = point_embedding[:, :, 1 : (1 + self.num_mask_tokens), :]

        image_embeddings = image_embeddings.transpose(2, 3).reshape(
            batch_size * point_batch_size, num_channels, height, width
        )

        upscaled_embedding = self.upscale_conv1(image_embeddings)
        upscaled_embedding = self.activation(self.upscale_layer_norm(upscaled_embedding))
        upscaled_embedding = self.activation(self.upscale_conv2(upscaled_embedding))

        upscaled_embedding_hq = self.mask_conv1(upscaled_embedding)
        upscaled_embedding_hq = self.activation(self.mask_norm(upscaled_embedding_hq))
        upscaled_embedding_hq = self.mask_conv2(upscaled_embedding_hq)

        if hq_features.shape[0] == 1:
            hq_features = hq_features.repeat(batch_size, 1, 1, 1)
        upscaled_embedding_hq = upscaled_embedding_hq + hq_features


        hyper_in_list = []
        for i in range(self.num_mask_tokens):
            if i < self.num_mask_tokens - 1:
                current_mlp = self.output_hypernetworks_mlps[i]
                hyper_in_list += [current_mlp(mask_tokens_out[:, :, i, :])]
            else:
                current_mlp = self.hf_mlp
                hyper_in_list += [current_mlp(mask_tokens_out[:, :, i, :])]

        

        hyper_in = torch.stack(hyper_in_list, dim=2)
        _, num_channels, height, width = upscaled_embedding.shape
        upscaled_embedding = upscaled_embedding.reshape(batch_size, point_batch_size, num_channels, height * width)
        upscaled_embedding_hq = upscaled_embedding_hq.reshape(batch_size, point_batch_size, num_channels, height * width)

        masks_sam = (hyper_in[:, :, :self.num_mask_tokens-1] @ upscaled_embedding).reshape(batch_size, point_batch_size, -1, height, width)
        masks_hq = (hyper_in[:, :, self.num_mask_tokens-1:] @ upscaled_embedding_hq).reshape(batch_size, point_batch_size, -1, height, width)
        masks = torch.cat([masks_sam, masks_hq], dim=2)


        iou_pred = self.iou_prediction_head(iou_token_out)




        if multimask_output:
            mask_slice = slice(1,self.num_mask_tokens-1)
            iou_pred = iou_pred[:, :, mask_slice]


            iou_pred, max_iou_idx = torch.max(iou_pred, dim=1)
            iou_pred = iou_pred.unsqueeze(1)


            masks_multi = masks[:, :, mask_slice, :, :]
            batch_indices = torch.arange(masks_multi.size(0)).unsqueeze(1).expand(-1, masks_multi.size(1))
            point_indices = torch.arange(masks_multi.size(1)).unsqueeze(0).expand(masks_multi.size(0), -1)
            masks_sam = masks_multi[batch_indices, point_indices, max_iou_idx].unsqueeze(1)
        else:
            mask_slice = slice(0, 1)
            iou_pred = iou_pred[:, :, mask_slice]
            masks_sam = masks[:, :, mask_slice, :, :]
            

        masks_hq = masks[:, :, slice(self.num_mask_tokens-1, self.num_mask_tokens), :, :]
        if hq_token_only:
            masks = masks_hq
        else:
            masks = masks_sam + masks_hq

        
        outputs = (masks, iou_pred)
        if output_attentions:
            outputs = outputs + (attentions,)
        else:
            outputs = outputs + (None,)

        return outputs
    

class SamHQPreTrainedModel(SamPreTrainedModel):
    pass


SAM_HQ_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`SamHQConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

SAM_HQ_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`SamHQProcessor`]. See [`SamHQProcessor.__call__`] for
            details.
        input_points (`torch.FloatTensor` of shape `(batch_size, num_points, 2)`):
            Input 2D spatial points, this is used by the prompt encoder to encode the prompt. Generally yields to much
            better results. The points can be obtained by passing a list of list of list to the processor that will
            create corresponding `torch` tensors of dimension 4. The first dimension is the image batch size, the
            second dimension is the point batch size (i.e. how many segmentation masks do we want the model to predict
            per input point), the third dimension is the number of points per segmentation mask (it is possible to pass
            multiple points for a single mask), and the last dimension is the x (vertical) and y (horizontal)
            coordinates of the point. If a different number of points is passed either for each image, or for each
            mask, the processor will create "PAD" points that will correspond to the (0, 0) coordinate, and the
            computation of the embedding will be skipped for these points using the labels.
        input_labels (`torch.LongTensor` of shape `(batch_size, point_batch_size, num_points)`):
            Input labels for the points, this is used by the prompt encoder to encode the prompt. According to the
            official implementation, there are 3 types of labels:

            - `1`: the point is a point that contains the object of interest
            - `0`: the point is a point that does not contain the object of interest
            - `-1`: the point corresponds to the background

            We added the label:

            - `-10`: the point is a padding point, thus should be ignored by the prompt encoder

            The padding labels should be automatically done by the processor.
        input_boxes (`torch.FloatTensor` of shape `(batch_size, num_boxes, 4)`):
            Input boxes for the points, this is used by the prompt encoder to encode the prompt. Generally yields to
            much better generated masks. The boxes can be obtained by passing a list of list of list to the processor,
            that will generate a `torch` tensor, with each dimension corresponding respectively to the image batch
            size, the number of boxes per image and the coordinates of the top left and botton right point of the box.
            In the order (`x1`, `y1`, `x2`, `y2`):

            - `x1`: the x coordinate of the top left point of the input box
            - `y1`: the y coordinate of the top left point of the input box
            - `x2`: the x coordinate of the bottom right point of the input box
            - `y2`: the y coordinate of the bottom right point of the input box

        input_masks (`torch.FloatTensor` of shape `(batch_size, image_size, image_size)`):
            SAM-HQ model also accepts segmentation masks as input. The mask will be embedded by the prompt encoder to
            generate a corresponding embedding, that will be fed later on to the mask decoder. These masks needs to be
            manually fed by the user, and they need to be of shape (`batch_size`, `image_size`, `image_size`).

        image_embeddings (`torch.FloatTensor` of shape `(batch_size, output_channels, window_size, window_size)`):
            Image embeddings, this is used by the mask decoder to generate masks and iou scores. For more memory
            efficient computation, users can first retrieve the image embeddings using the `get_image_embeddings`
            method, and then feed them to the `forward` method instead of feeding the `pixel_values`.
        multimask_output (`bool`, *optional*):
            In the original implementation and paper, the model always outputs 3 masks per image (or per point / per
            bounding box if relevant). However, it is possible to just output a single mask, that corresponds to the
            "best" mask, by specifying `multimask_output=False`.
        hq_token_only (`bool`, *optional*, defaults to `False`):
            Whether to use only the HQ token path for mask generation. When False, combines both standard and HQ paths.
            This is specific to SAM-HQ's architecture.
        attention_similarity (`torch.FloatTensor`, *optional*):
            Attention similarity tensor, to be provided to the mask decoder for target-guided attention in case the
            model is used for personalization as introduced in [PerSAM](https://arxiv.org/abs/2305.03048).
        target_embedding (`torch.FloatTensor`, *optional*):
            Embedding of the target concept, to be provided to the mask decoder for target-semantic prompting in case
            the model is used for personalization as introduced in [PerSAM](https://arxiv.org/abs/2305.03048).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        interm_embeddings (`List[torch.FloatTensor]`, *optional*):
            Intermediate embeddings from vision encoder's non-windowed blocks, used by SAM-HQ for enhanced mask quality.
            Required when providing pre-computed image_embeddings instead of pixel_values.
"""


@add_start_docstrings(
    "Segment Anything Model HQ (SAM-HQ) for generating masks,given an input image and",
    " optional 2D location and bounding boxes.",
    SAM_HQ_START_DOCSTRING,
)
class SamHQModel(SamHQPreTrainedModel):
    _tied_weights_keys = ["prompt_encoder.shared_embedding.positional_embedding"]


    def __init__(self, config):
        super().__init__(config)
        self.shared_image_embedding = SamHQPositionalEmbedding(config.vision_config)

        self.vision_encoder = SamHQVisionEncoder(config.vision_config)
        self.prompt_encoder = SamHQPromptEncoder(config.prompt_encoder_config, self.shared_image_embedding)
        self.mask_decoder = SamHQMaskDecoder(config.mask_decoder_config)

        self.post_init()


    def get_input_embeddings(self):
        return self.vision_encoder.get_input_embeddings()
    
    def get_image_wide_positional_embeddings(self):
        size = self.config.prompt_encoder_config.image_embedding_size
        target_device = self.shared_image_embedding.positional_embedding.device
        target_dtype = self.shared_image_embedding.positional_embedding.dtype
        grid = torch.ones((size, size), device=target_device, dtype=target_dtype)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / size
        x_embed = x_embed / size

        positional_embedding = self.shared_image_embedding(torch.stack([x_embed, y_embed], dim=-1))
        return positional_embedding.permute(2, 0, 1).unsqueeze(0)
    


    @torch.no_grad()
    def get_image_embeddings(
        self,
        pixel_values,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,

    ):
        r"""
        Returns the image embeddings by passing the pixel values through the vision encoder.

        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
                Input pixel values
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.

        """
        vision_output = self.vision_encoder(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        image_embeddings = vision_output[0]
        intermediate_embeddings = vision_output[1]

        return image_embeddings, intermediate_embeddings
    

    @torch.no_grad()
    def get_prompt_embeddings(
        self,
        input_points: Optional[torch.FloatTensor]= None,
        input_labels: Optional[torch.LongTensor]= None,
        input_boxes: Optional[torch.FloatTensor]= None,
        input_masks: Optional[torch.LongTensor]= None,
    ):
        r"""
        Returns the prompt embeddings by passing the input points, labels, boxes and masks through the prompt encoder.

        Args:
            input_points (`torch.FloatTensor` of shape `(batch_size, point_batch_size, num_points_per_image, 2)`):
                Optional input points for the prompt encoder. The padding of the point is automatically done by the
                processor. `point_batch_size` refers to the number of masks that we want the model to predict per
                point. The model will output `point_batch_size` times 3 masks in total.
            input_labels (`torch.LongTensor` of shape `(batch_size, point_batch_size, num_points_per_image)`):
                Optional input labels for the prompt encoder. The padding of the labels is automatically done by the
                processor, or can be fed by the user.
            input_boxes (`torch.FloatTensor` of shape `(batch_size, num_boxes_per_image, 4)`):
                Optional input boxes for the prompt encoder. The padding of the boxes is automatically done by the
                processor. users can also pass manually the input boxes.
            input_masks (`torch.LongTensor` of shape `(batch_size, image_size, image_size)`):
                Optional input masks for the prompt encoder.
        """
        prompt_output = self.prompt_encoder(
            input_points=input_points,
            input_labels=input_labels,
            input_boxes=input_boxes,
            input_masks=input_masks,
        )
        return prompt_output
    

    @add_start_docstrings_to_model_forward(SAM_HQ_INPUTS_DOCSTRING)
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        input_points: Optional[torch.FloatTensor] = None,
        input_labels: Optional[torch.LongTensor] = None,
        input_boxes: Optional[torch.FloatTensor] = None,
        input_masks: Optional[torch.LongTensor] = None,
        image_embeddings: Optional[torch.FloatTensor] = None,
        multimask_output: bool = True,
        hq_token_only: bool = False,
        attention_similarity: Optional[torch.FloatTensor] = None,
        target_embedding: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        interm_embeddings: Optional[List[torch.FloatTensor]] = None,
        **kwargs,
    ) -> List[Dict[str, torch.Tensor]]:
        r"""
        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoModel, AutoProcessor

        >>> model = AutoModel.from_pretrained("sushmanth/sam_hq_vit_b")
        >>> processor = AutoProcessor.from_pretrained("sushmanth/sam_hq_vit_b")

        >>> img_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/sam-car.png"
        >>> raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
        >>> input_points = [[[400, 650]]]  # 2D location of a window on the car
        >>> inputs = processor(images=raw_image, input_points=input_points, return_tensors="pt")

        >>> # Get high-quality segmentation mask
        >>> outputs = model(**inputs)

        >>> # For high-quality mask only
        >>> outputs = model(**inputs, hq_token_only=True)

        >>> # Postprocess masks
        >>> masks = processor.post_process_masks(
        ...     outputs.pred_masks, inputs["original_sizes"], inputs["reshaped_input_sizes"]
        ... )
        ```
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None and image_embeddings is None:
            raise ValueError("Either pixel_values or image_embeddings must be provided.")

        if pixel_values is not None and image_embeddings is not None:
            raise ValueError("Only one of pixel_values and image_embeddings can be provided.")

        if input_points is not None and len(input_points.shape) != 4:
            raise ValueError(
                "The input_points must be a 4D tensor. Of shape `batch_size`, `point_batch_size`, `nb_points_per_image`, `2`."
                f" got {input_points.shape}."
            )
        
        if input_boxes is not None and len(input_boxes.shape) != 3:
            raise ValueError(
                "The input_boxes must be a 3D tensor. Of shape `batch_size`, `nb_boxes`, `4`."
                f" got {input_boxes.shape}."
            )

        # Add validation for point and box batch sizes
        if input_points is not None and input_boxes is not None:
            point_batch_size = input_points.shape[1]
            box_batch_size = input_boxes.shape[1]
            if point_batch_size != box_batch_size:
                raise ValueError(
                    "You should provide as many bounding boxes as input points per box. Got {} and {}.".format(
                        point_batch_size, box_batch_size
                    )
                )

        image_positional_embeddings = self.get_image_wide_positional_embeddings()
        # repeat with batch size
        batch_size = pixel_values.shape[0] if pixel_values is not None else image_embeddings.shape[0]
        image_positional_embeddings = image_positional_embeddings.repeat(batch_size, 1, 1, 1)

        vision_attentions = None
        vision_hidden_states = None
        
        if pixel_values is not None:
            vision_outputs = self.vision_encoder(
                pixel_values,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            
            if return_dict:
                image_embeddings = vision_outputs.last_hidden_state
                interm_embeddings = vision_outputs.intermediate_embeddings
                if output_hidden_states:
                    vision_hidden_states = vision_outputs.hidden_states
                if output_attentions:
                    vision_attentions = vision_outputs.attentions
            else:
                image_embeddings = vision_outputs[0]
                interm_embeddings = vision_outputs[1]
                if output_hidden_states:
                    vision_hidden_states = vision_outputs[2]
                if output_attentions:
                    vision_attentions = vision_outputs[-1]

        if input_points is not None and input_labels is None:
            input_labels = torch.ones_like(input_points[:, :, :, 0], dtype=torch.int, device=input_points.device)

        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            input_points=input_points,
            input_labels=input_labels,
            input_boxes=input_boxes,
            input_masks=input_masks,
        )

        # Predict masks
        low_res_masks, iou_predictions, mask_decoder_attentions = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_positional_embeddings=image_positional_embeddings,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            hq_token_only=hq_token_only,
            interm_embeddings=interm_embeddings,
            attention_similarity=attention_similarity,
            target_embedding=target_embedding,
            output_attentions=output_attentions,
        )

        if not return_dict:
            output = (iou_predictions, low_res_masks)
            if output_hidden_states:
                output = output + (vision_hidden_states,)

            if output_attentions:
                output = output + (vision_attentions, mask_decoder_attentions)
            return output

        return SamHQImageSegmentationOutput(
            iou_scores=iou_predictions,
            pred_masks=low_res_masks,
            vision_hidden_states=vision_hidden_states,
            vision_attentions=vision_attentions,
            mask_decoder_attentions=mask_decoder_attentions,
        )
    


    

