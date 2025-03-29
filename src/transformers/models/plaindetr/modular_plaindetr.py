import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math
from ..deformable_detr.modeling_deformable_detr import DeformableDetrSinePositionEmbedding, DeformableDetrLearnedPositionEmbedding



class PlainDetrSinePositionEmbedding(DeformableDetrSinePositionEmbedding):
    pass

class PlainDetrLearnedPositionEmbedding(DeformableDetrLearnedPositionEmbedding):
    pass

def build_position_encoding(config):
    if config.position_embedding_type == "sine":
        position_embedding = PlainDetrSinePositionEmbedding(n_steps, normalize=True)
    elif config.position_embedding_type == "learned":
        position_embedding = PlainDetrLearnedPositionEmbedding(n_steps)
    elif config.position_embedding_type == "sine":
        position_embedding = PlainDetrSinePositionEmbedding(n_steps, normalize=False)
    else:
        raise ValueError(f"Unknown position embedding type: {config.position_embedding_type}")
    
    return position_embedding


class GlobalRpeDecompAttentionWithPositionEmbedding(nn.Module):
    """
    Global cross-attention module used in Global RPE Decomposition model.
    
    This module implements cross-attention between query embeddings and 
    flattened feature maps, incorporating position information.
    
    Args:
        dim (`int`): The dimension of the model.
        num_heads (`int`): Number of attention heads.
        qkv_bias (`bool`, *optional*, defaults to `True`):
            Whether to add a bias to the query, key, value projections.
        qk_scale (`float`, *optional*, defaults to `None`):
            Scale factor for query key attention scores. If None, uses head_dim ** -0.5 as default.
        attn_drop (`float`, *optional*, defaults to 0.0):
            Dropout probability for attention weights.
        proj_drop (`float`, *optional*, defaults to 0.0):
            Dropout probability for projection weights.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # Define projections
        self.query_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.key_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.value_proj = nn.Linear(dim, dim, bias=qkv_bias)
        
        # Define dropouts and output projection
        self.attention_dropout = nn.Dropout(attn_drop)
        self.output_proj = nn.Linear(dim, dim)
        self.output_dropout = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self,
        query_vectors: torch.Tensor,
        key_input_flatten: torch.Tensor,
        value_input_flatten: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query_vectors: Query embeddings of shape (batch_size, num_queries, dim)
            key_input_flatten: Key input of shape (batch_size, seq_len, dim)
            value_input_flatten: Value input of shape (batch_size, seq_len, dim)
            attention_mask: Optional attention mask of shape (batch_size, seq_len)
            
        Returns:
            torch.Tensor: Output features of shape (batch_size, num_queries, dim)
        """
        # Process keys and values
        batch_size, seq_len, embedding_dim = key_input_flatten.shape
        keys = self.key_proj(key_input_flatten).reshape(
            batch_size, seq_len, self.num_heads, embedding_dim // self.num_heads
        )
        keys = keys.permute(0, 2, 1, 3)  # B, nheads, seq_len, head_dim
        
        values = self.value_proj(value_input_flatten).reshape(
            batch_size, seq_len, self.num_heads, embedding_dim // self.num_heads
        )
        values = values.permute(0, 2, 1, 3)  # B, nheads, seq_len, head_dim
        
        # Process queries
        batch_size, seq_len, embedding_dim = query_vectors.shape
        queries = self.query_proj(query_vectors).reshape(
            batch_size, seq_len, self.num_heads, embedding_dim // self.num_heads
        )
        queries = queries.permute(0, 2, 1, 3)  # B, nheads, nQ, head_dim
        queries = queries * self.scale

        # Compute attention scores
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1))
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask[:, None, None] * -100
            
        # Clip values to prevent overflow/underflow
        dtype = attention_scores.dtype
        fmin, fmax = torch.finfo(dtype).min, torch.finfo(dtype).max
        attention_scores = torch.clamp(attention_scores, min=fmin, max=fmax)
            
        # Apply softmax and dropout
        attention_probs = self.softmax(attention_scores)
        attention_probs = self.attention_dropout(attention_probs)
        
        # Compute weighted sum
        context_layer = torch.matmul(attention_probs, values)
        context_layer = context_layer.permute(0, 2, 1, 3).reshape(
            batch_size, seq_len, embedding_dim
        )
        
        # Apply output projection and dropout
        output = self.output_proj(context_layer)
        output = self.output_dropout(output)
        
        return output


class GlobalRpeDecompCrossAttention(nn.Module):
    """
    Global cross-attention module with relative position encoding for Global RPE Decomposition model.
    
    This module implements cross-attention between query embeddings and feature maps,
    incorporating relative position information between queries and feature map positions.
    
    Args:
        dim (`int`): The dimension of the model.
        num_heads (`int`): Number of attention heads.
        qkv_bias (`bool`, *optional*, defaults to `True`):
            Whether to add a bias to the query, key, value projections.
        qk_scale (`float`, *optional*, defaults to `None`):
            Scale factor for query key attention scores. If None, uses head_dim ** -0.5 as default.
        attn_drop (`float`, *optional*, defaults to 0.0):
            Dropout probability for attention weights.
        proj_drop (`float`, *optional*, defaults to 0.0):
            Dropout probability for projection weights.
        rpe_hidden_dim (`int`, *optional*, defaults to 512):
            Hidden dimension for relative position encoding MLPs.
        rpe_type (`str`, *optional*, defaults to 'linear'):
            Type of relative position encoding to use.
        feature_stride (`int`, *optional*, defaults to 16):
            Stride for feature map positions.
        reparam (`bool`, *optional*, defaults to False):
            Whether reference points are already in absolute coordinates.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        rpe_hidden_dim: int = 512,
        rpe_type: str = 'linear',
        feature_stride: int = 16,
        reparam: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.rpe_type = rpe_type
        self.feature_stride = feature_stride
        self.reparam = reparam

        # Define MLPs for relative position encoding
        self.cpb_mlp1 = nn.Sequential(
            nn.Linear(2, rpe_hidden_dim, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(rpe_hidden_dim, num_heads, bias=False)
        )
        self.cpb_mlp2 = nn.Sequential(
            nn.Linear(2, rpe_hidden_dim, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(rpe_hidden_dim, num_heads, bias=False)
        )

        # Define projections
        self.query_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.key_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.value_proj = nn.Linear(dim, dim, bias=qkv_bias)
        
        # Define dropouts and output projection
        self.attention_dropout = nn.Dropout(attn_drop)
        self.output_proj = nn.Linear(dim, dim)
        self.output_dropout = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        reference_points: torch.Tensor,
        key_value_states: torch.Tensor,
        spatial_shapes: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: Query embeddings of shape (batch_size, num_queries, dim)
            reference_points: Reference points of shape (batch_size, num_queries, 1, 4)
            key_value_states: Feature map of shape (batch_size, h*w, dim)
            spatial_shapes: Feature map shapes of shape (1, 2)
            attention_mask: Optional attention mask of shape (batch_size, h*w)
            
        Returns:
            torch.Tensor: Output features of shape (batch_size, num_queries, dim)
        """
        assert spatial_shapes.size(0) == 1, 'This is designed for single-scale decoder.'
        h, w = spatial_shapes[0]
        stride = self.feature_stride

        # Convert center-size format to corner format
        ref_pts = torch.cat([
            reference_points[:, :, :, :2] - reference_points[:, :, :, 2:] / 2,
            reference_points[:, :, :, :2] + reference_points[:, :, :, 2:] / 2,
        ], dim=-1)  # B, nQ, 1, 4
        
        # Scale reference points to absolute coordinates if not already done
        if not self.reparam:
            ref_pts[..., 0::2] *= (w * stride)
            ref_pts[..., 1::2] *= (h * stride)
            
        # Create position grid
        device = hidden_states.device
        pos_x = torch.linspace(0.5, w - 0.5, w, dtype=torch.float32, device=device)
        pos_x = pos_x[None, None, :, None] * stride  # 1, 1, w, 1
        pos_y = torch.linspace(0.5, h - 0.5, h, dtype=torch.float32, device=device)
        pos_y = pos_y[None, None, :, None] * stride  # 1, 1, h, 1

        # Compute relative position encodings
        if self.rpe_type == 'abs_log8':
            delta_x = ref_pts[..., 0::2] - pos_x  # B, nQ, w, 2
            delta_y = ref_pts[..., 1::2] - pos_y  # B, nQ, h, 2
            log2_8 = math.log2(8)
            delta_x = torch.sign(delta_x) * torch.log2(torch.abs(delta_x) + 1.0) / log2_8
            delta_y = torch.sign(delta_y) * torch.log2(torch.abs(delta_y) + 1.0) / log2_8
        elif self.rpe_type == 'linear':
            delta_x = ref_pts[..., 0::2] - pos_x  # B, nQ, w, 2
            delta_y = ref_pts[..., 1::2] - pos_y  # B, nQ, h, 2
        else:
            raise NotImplementedError(
                f"Relative position encoding type {self.rpe_type} not implemented"
            )

        # Process relative positions through MLPs
        rpe_x = self.cpb_mlp1(delta_x)  # B, nQ, w, nheads
        rpe_y = self.cpb_mlp2(delta_y)  # B, nQ, h, nheads
        rpe = (rpe_x[:, :, None] + rpe_y[:, :, :, None]).flatten(2, 3)  # B, nQ, h*w, nheads
        rpe = rpe.permute(0, 3, 1, 2)  # B, nheads, nQ, h*w

        # Process keys and values
        batch_size, seq_len, embedding_dim = key_value_states.shape
        keys = self.key_proj(key_value_states).reshape(
            batch_size, seq_len, self.num_heads, embedding_dim // self.num_heads
        )
        keys = keys.permute(0, 2, 1, 3)  # B, nheads, h*w, head_dim
        
        values = self.value_proj(key_value_states).reshape(
            batch_size, seq_len, self.num_heads, embedding_dim // self.num_heads
        )
        values = values.permute(0, 2, 1, 3)  # B, nheads, h*w, head_dim

        # Process queries
        batch_size, seq_len, embedding_dim = hidden_states.shape
        queries = self.query_proj(hidden_states).reshape(
            batch_size, seq_len, self.num_heads, embedding_dim // self.num_heads
        )
        queries = queries.permute(0, 2, 1, 3)  # B, nheads, nQ, head_dim
        queries = queries * self.scale

        # Compute attention scores
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1))
        attention_scores = attention_scores + rpe

        # Apply attention mask if provided
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask[:, None, None] * -100

        # Clip values to prevent overflow/underflow
        dtype = attention_scores.dtype
        fmin, fmax = torch.finfo(dtype).min, torch.finfo(dtype).max
        attention_scores = torch.clamp(attention_scores, min=fmin, max=fmax)

        # Apply softmax and dropout
        attention_probs = self.softmax(attention_scores)
        attention_probs = self.attention_dropout(attention_probs)

        # Compute weighted sum
        context_layer = torch.matmul(attention_probs, values)
        context_layer = context_layer.permute(0, 2, 1, 3).reshape(
            batch_size, seq_len, embedding_dim
        )
        
        # Apply output projection and dropout
        output = self.output_proj(context_layer)
        output = self.output_dropout(output)

        return output

