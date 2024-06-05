import torch
import torch.nn as nn
import torch.nn.functional as F

from segmentation_models_pytorch.base import modules as md

#==============================================================================

# # Series attention 
# class DecoderBlock(nn.Module):
#     def __init__(
#         self,
#         in_channels,
#         skip_channels,
#         out_channels,
#         use_batchnorm=True,
#         attention_type=None,
#     ):
#         super().__init__()
#         self.conv1 = md.Conv2dReLU(
#             in_channels + skip_channels,
#             out_channels,
#             kernel_size=3,
#             padding=1,
#             use_batchnorm=use_batchnorm,
#         )
        
#         self.attention1 = md.Attention(attention_type, in_channels=in_channels + skip_channels)
#         # self.attention1 = md.SCSEModule0(in_channels=in_channels + skip_channels, strategy='addition')
        
#         self.conv2 = md.Conv2dReLU(
#             out_channels,
#             out_channels,
#             kernel_size=3,
#             padding=1,
#             use_batchnorm=use_batchnorm,
#         )
        
#         self.attention2 = md.Attention(attention_type, in_channels=out_channels)
#         # self.attention2 = md.SCSEModule0(in_channels=out_channels, strategy='addition')

#     def forward(self, x, skip=None):
#         x = F.interpolate(x, scale_factor=2, mode="nearest")
#         if skip is not None:
#             x = torch.cat([x, skip], dim=1)
#             x = self.attention1(x)
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = self.attention2(x)
#         return x
    
#==============================================================================

#==============================================================================

# Parallel attention v1: diffent attention when no skip connection
class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        use_batchnorm=True,
        attention_type=None,
    ):
        super().__init__()
        self.conv1 = md.Conv2dReLU(
            # 2 * (in_channels + skip_channels), # -------->> use for concatenation <<----------
            in_channels + skip_channels, # -------->> use for addition <<----------
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        
        # self.attention1 = md.Attention(attention_type, in_channels=in_channels + skip_channels)
        self.attention1 = md.SCSEModule0(in_channels=in_channels + skip_channels, strategy='maxout')
        
        self.conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        
        # self.attention2 = md.Attention(attention_type, in_channels=out_channels)
        self.attention2 = md.SCSEModule0(in_channels=in_channels + skip_channels, strategy='addition')

    def forward(self, x, skip=None, strategy='add'):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x1 = self.attention1(x) # first attention         
            x2 = self.attention2(x) # 2nd attention

            
            # Experimental
            # x1 = self.conv1(x1) # --> experimental
            # x2 = self.conv1(x2) # --> experimental            
            
            
            if strategy == 'cat': x3 = torch.cat([x1, x2], dim=1) # concatenate two attentions   
            elif strategy == 'add': x3 = x1 + x2
            
        else:
            x2 = self.attention2(x) # 2nd attention
            
            # # Experimental
            # x = self.conv1(x)
            # x2 = self.conv1(x2)
            
            
            if strategy == 'cat': x3 = torch.cat([x, x2], dim=1) # concatenate x and x2
            elif strategy == 'add': x3 = x + x2
        
        x3 = self.conv1(x3)
        
        # # Experimental
        # x3 = self.conv2(x3)
        
        # x = self.conv2(x)
        # x = self.attention2(x)
        return x3

#---------------------------------------------------------------

# # Parallel attention v2: same attention for both skip and no-skip connection
# class DecoderBlock(nn.Module):
#     def __init__(
#         self,
#         in_channels,
#         skip_channels,
#         out_channels,
#         use_batchnorm=True,
#         attention_type=None,
#     ):
#         super().__init__()
#         self.conv1 = md.Conv2dReLU(
#             # 2 * (in_channels + skip_channels), # -------->> use for concatenation <<----------
#             in_channels + skip_channels, # -------->> use for addition <<----------
#             out_channels,
#             kernel_size=3,
#             padding=1,
#             use_batchnorm=use_batchnorm,
#         )
        
#         # self.attention1 = md.Attention(attention_type, in_channels=in_channels + skip_channels)
#         self.attention1 = md.SCSEModule0(in_channels=in_channels + skip_channels, strategy='maxout')
        
#         self.conv2 = md.Conv2dReLU(
#             in_channels,
#             out_channels,
#             kernel_size=3,
#             padding=1,
#             use_batchnorm=use_batchnorm,
#         )
        
#         # self.attention2 = md.Attention(attention_type, in_channels=out_channels)
#         self.attention2 = md.SCSEModule0(in_channels=in_channels + skip_channels, strategy='addition')

#     def forward(self, x, skip=None, strategy='add'):
#         x = F.interpolate(x, scale_factor=2, mode="nearest")
#         if skip is not None:
#             x = torch.cat([x, skip], dim=1)
#             x1 = self.attention1(x) # first attention         
#             x2 = self.attention2(x) # 2nd attention
        
#             if strategy == 'cat': x3 = torch.cat([x1, x2], dim=1) # concatenate two attentions   
#             elif strategy == 'add': x3 = x1 + x2
            
#             x3 = self.conv1(x3)
            
#         else:
#             x1 = self.attention1(x) # first attention         
#             x2 = self.attention2(x) # 2nd attention
            
#             if strategy == 'cat': x3 = torch.cat([x1, x2], dim=1) # concatenate two attentions   
#             elif strategy == 'add': x3 = x1 + x2
            
#             x3 = self.conv2(x3)

#         return x3
    
#==============================================================================


# # Single attention (v1: as per the original package)
# class DecoderBlock(nn.Module):
#     def __init__(
#         self,
#         in_channels,
#         skip_channels,
#         out_channels,
#         use_batchnorm=True,
#         attention_type=None,
#     ):
#         super().__init__()
#         self.conv1 = md.Conv2dReLU(
#             in_channels + skip_channels,
#             out_channels,
#             kernel_size=3,
#             padding=1,
#             use_batchnorm=use_batchnorm,
#         )
        
#         # self.attention1 = md.Attention(attention_type, in_channels=in_channels + skip_channels)
#         self.attention1 = md.SCSEModule0(in_channels=in_channels + skip_channels, strategy='maxout')
        
#         self.conv2 = md.Conv2dReLU(
#             out_channels,
#             out_channels,
#             kernel_size=3,
#             padding=1,
#             use_batchnorm=use_batchnorm,
#         )
        
#         self.attention2 = md.Attention(attention_type, in_channels=out_channels)
#         # self.attention2 = md.SCSEModule0(in_channels=out_channels, strategy='addition')

#     def forward(self, x, skip=None):
#         x = F.interpolate(x, scale_factor=2, mode="nearest")
#         if skip is not None:
#             x = torch.cat([x, skip], dim=1)
#             x = self.attention1(x)
#         x = self.conv1(x)
#         # x = self.conv2(x)
#         # x = self.attention2(x)
#         return x

#==============================================================================

# # Single attention (v2: modified version (as per the original paper))
# class DecoderBlock(nn.Module):
#     def __init__(
#         self,
#         in_channels,
#         skip_channels,
#         out_channels,
#         use_batchnorm=True,
#         attention_type=None,
#     ):
#         super().__init__()
#         self.conv1 = md.Conv2dReLU(
#             in_channels + skip_channels,
#             out_channels,
#             kernel_size=3,
#             padding=1,
#             use_batchnorm=use_batchnorm,
#         )
        
        
#         self.conv2 = md.Conv2dReLU(
#             out_channels,
#             out_channels,
#             kernel_size=3,
#             padding=1,
#             use_batchnorm=use_batchnorm,
#         )
        
#         self.attention = md.SCSEModule0(in_channels=out_channels, strategy='addition')

#     def forward(self, x, skip=None):
#         x = F.interpolate(x, scale_factor=2, mode="nearest")
#         if skip is not None:
#             x = torch.cat([x, skip], dim=1)
#             # x = self.attention1(x)
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = self.attention(x)
#         return x

#==============================================================================

class CenterBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        conv1 = md.Conv2dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        super().__init__(conv1, conv2)


class UnetDecoder(nn.Module):
    def __init__(
        self,
        encoder_channels,
        decoder_channels,
        n_blocks=5,
        use_batchnorm=True,
        attention_type=None,
        center=False,
    ):
        super().__init__()

        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )

        # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[1:]
        # reverse channels to start from head of encoder
        encoder_channels = encoder_channels[::-1]

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels

        if center:
            self.center = CenterBlock(head_channels, head_channels, use_batchnorm=use_batchnorm)
        else:
            self.center = nn.Identity()

        # combine decoder keyword arguments
        kwargs = dict(use_batchnorm=use_batchnorm, attention_type=attention_type)
        blocks = [
            DecoderBlock(in_ch, skip_ch, out_ch, **kwargs)
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, *features):
               
        features = features[1:]  # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder

        head = features[0]
        skips = features[1:]

        x = self.center(head)
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)

        return x
