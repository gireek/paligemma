from turtle import forward
from typing import Optional, Tuple
import torch
import torch.nn as nn
from math import sqrt

import torch.nn.Functional as F


# some copied Config from Umar
class SiglipVisionConfig:

    def __init__(
        self,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_channels=3,
        image_size=224,
        patch_size=16,
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        num_image_tokens: int = None,
        **kwargs
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size # in ffn inside
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels # 3 channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.num_image_tokens = num_image_tokens



# STEP 1 - Convolution + postional embedding implementation
# Construct SigLipVisionEmbeddings which takes pixel_values in the form 
# [b, c, h, w] and converts to patch wise embeddings

# class SigLipVisionEmbeddings(nn.Module):
#     def __init__(self, config: SiglipVisionConfig):
#         #####################################

class SiglipVisionEmbeddings(nn.Module):

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.conv = nn.Conv2d(
            in_channels=3,
            out_channels=config.hidden_size,
            kernel_size=config.patch_size,
            stride= config.patch_size ,
            padding= "valid" #ensure no padding added
        )

        self.num_patches = (config.image_size // config.patch_size) ** 2

        # position embedding is a lookup table going from number of patches to hidden_size dimensional vector
        self.position_embeddings = nn.Embedding(self.num_patches, config.hidden_size)


    # if x is of the shape [16, 3, 224, 224]
    # after conv shape is [16, 768, 14, 14] where 14 is number of patches per row making it 14*14 total patches
    # after flatten shape is [16, 768, 196]
    # transpose to match direction with position embeddings
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = torch.flatten(x, start_dim = 2) # flatten everything after dim 2

        # interchange bcz we want to give sequence of patch embeddings
        x = x.transpose(1, 2)
        x += self.position_embeddings(torch.tensor(range(self.num_patches)))

        return x


# once we have the embeddings from the initial step we can
# pass to blocks of (layer norm + self attention + MLP) 
# to get contextualized embeddings


# STEP 2 - Multi layer perceptron implementation
# Construct SiglipMLP which takes embeddings in the form 
# [b, num_patches, embed_dim] and converts to the same shape just smarter :p

# class SiglipMLP(nn.Module):
#     def __init__(self, config: SiglipVisionConfig):
#         #####################################

class SiglipMLP(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        # first stretch the embed_dim to something bigger like intermediate_size
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.gelu(self.fc1(x)))


#  STEP 3 - multi head attention
#  Construct SiglipAttention which takes embeddings in the form 
# [b, num_patches, embed_dim] and converts to the same shape just CONTEXTUALIZED!

# class SiglipAttention(nn.Module):
#     def __init__(self, config: SiglipVisionConfig):
#         super().__init__()
#         ########################


class SigLipAttention(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()

        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size/self.num_heads
        self.dropout = config.attention_dropout

        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size)


    def forward(self, x: torch.Tensor) -> torch.Tensor:

        batch_size, num_patches, embed_dim = x.size()

        k = self.k_proj(x)
        q = self.q_proj(x)
        v = self.v_proj(x)

        # why transpose below? each head shhould be able to 
        # visualize everything separately and in parallel
        q = q.view(batch_size, num_patches, self.num_heads, self.head_dim).transpose(1,2)
        k = k.view(batch_size, num_patches, self.num_heads, self.head_dim).transpose(1,2)
        v = v.view(batch_size, num_patches, self.num_heads, self.head_dim).transpose(1,2)

        # all are shaped [batch_size, self.num_heads, num_patches, self.head_dim] as of now

        # transpose required to get num_patches x num_patches square in the
        #  end eliminating head_dim
        attn_weights = (q @ k.transpose(2,3))/sqrt(self.embed_dim)
        attn_weights = F.softmax(attn_weights, dim = -1 )
        attn_weights = F.dropout(attn_weights, dropout = self.dropout)
        # take a weighted sum of value vectors
        attn_output = attn_weights @ v
        attn_output = attn_output.transpose(1,2).contiguous()
        attn_output = attn_output.view(batch_size, num_patches, self.embed_dim)

        # to mix between heads
        final = self.o_proj(attn_output)

        return final


#  STEP 4 - Make a layer 
#  Construct SiglipEncoderLayer which takes embeddings and converts to the same shape after going through loops
#  of normalization, attention, skip connection, normalization, MLP and another skip connection going forward

# class SiglipEncoderLayer(nn.Module):
#     def __init__(self, config: SiglipVisionConfig):
#         super().__init__()
#         ########################


class SiglipEncoderLayer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()

        self.layer_norm1 = nn.LayerNorm(config.hidden_size)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size)
        self.self_attn = SigLipAttention(config)
        self.mlp = SiglipMLP(config)

    def forward(self, x):

        normalized_x = self.norm1(x)
        att_x = self.att_block(normalized_x)
        att_x += x

        normalized_att_x = self.norm2(att_x)
        mlp_x = self.mlp(normalized_att_x)
        mlp_x += att_x

        return mlp_x



#  STEP 5 - Make a layer 
#  Construct SiglipEncoderLayer which takes embeddings and converts to the same shape after going through loops
#  of normalization, attention, skip connection, normalization, MLP and another skip connection going forward

# class SiglipEncoderLayer(nn.Module):
#     def __init__(self, config: SiglipVisionConfig):
#         super().__init__()
#         ########################


