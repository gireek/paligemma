from typing import Optional, Tuple
import torch
import torch.nn as nn

# some copied Config
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
#         super().__init__()
#         #####################################

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#       ###################################


class SigLipVisionEmbeddings(nn.Module):

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



