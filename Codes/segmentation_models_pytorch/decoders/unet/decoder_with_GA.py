import torch
import torch.nn as nn
import torch.nn.functional as F

from segmentation_models_pytorch.base import modules as md

# =============================================================================
def activations(activation: str):
    ''' Choose the activation function '''
    
    if activation == 'relu': return nn.ReLU(inplace=True)
    elif activation == 'leaky': return nn.LeakyReLU(negative_slope=0.1)
    elif activation == 'elu': return nn.ELU()
    elif activation == 'sigmoid': return nn.Sigmoid()
    elif activation == 'softmax': return nn.Softmax(dim=1)
    else: raise ValueError('Wrong keyword for activation')

def normalization(norm: str, n_channel):
    ''' Choose type of normalization '''
    
    if norm == 'batch': return nn.BatchNorm2d(n_channel)
    elif norm == 'instance': return nn.InstanceNorm2d(n_channel)
    elif norm == None: pass # do nothing
    else: raise ValueError('Wrong keyword for normalization') 

# Weight initialization with keras default value
def weight_init_keras_default(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(m.bias)

# Weight initialization with truncated normal
def weight_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.trunc_normal_(m.weight, std=0.1)
        m.bias.data.zero_()
        
def gating_signal(in_channel, out_channel, activation: str, norm=None):
    gate = nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding='same'),
        normalization(norm, out_channel),
        activations(activation),
        )

    # gate.apply(weight_init_keras_default)
    
    return gate

class attention_block(nn.Module):
    def __init__(self, F_int, skip_channels): # previously shape, now using skip_channels
        super(attention_block, self).__init__()
        
        self.F_int = F_int
        
        self.theta_x = nn.Conv2d(F_int, F_int, kernel_size=2, stride=2, padding=0, bias=True)
        # self.theta_x.apply(weight_init_keras_default)
        
        self.phi_g = nn.Conv2d(F_int, F_int, kernel_size=1, padding='same')
        # self.phi_g.apply(weight_init_keras_default)
        
        self.upsample_g = nn.ConvTranspose2d(F_int, F_int, kernel_size=3, stride=1, padding=1)
        # self.upsample_g.apply(weight_init_keras_default)
        
        # Here is the concatenation step
        
        self.act_xg = nn.ReLU(inplace=True)
        
        self.psi = nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True)
        # self.psi.apply(weight_init_keras_default)
        
        self.sigmoid_xg = nn.Sigmoid()
        
        self.upsample_psi = nn.Upsample(scale_factor=2)
        
        # Here is repeat elements. upsample_psi has 1 ch only. Repeat it F_int times.
        
        # Here is multiplication between x and upsample_psi
        
        self.result = nn.Conv2d(F_int, skip_channels, kernel_size=1, padding='same')
        # self.result.apply(weight_init_keras_default)
        
        self.result_bn = nn.BatchNorm2d(skip_channels)
         
    def forward(self, g, x):
        theta_x = self.theta_x(x)
        phi_g = self.phi_g(g)
        upsample_g = self.upsample_g(phi_g)        
        concat_xg = upsample_g + theta_x
        act_xg = self.act_xg(concat_xg)
        psi = self.psi(act_xg)
        sigmoid_xg = self.sigmoid_xg(psi)
        upsample_psi = self.upsample_psi(sigmoid_xg)
        upsample_psi = torch.repeat_interleave(upsample_psi, self.F_int, dim=1)
        y = upsample_psi * x
        result = self.result(y)
        result_bn = self.result_bn(result)
        
        return result_bn
# =============================================================================

#==============================================================================
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
        
        self.skip_channels = skip_channels # added for MiT
        
        self.attention_type = attention_type
        
        self.gate = gating_signal(in_channels, skip_channels, 'relu', 'batch')
        
        self.gated_attention = attention_block(skip_channels, skip_channels)
        
        self.dropout = nn.Dropout2d(p=0.1) 
        
        # print('**************************', skip_channels, '\n')
        # print('**************************', in_channels, '\n')
        # print('**************************', out_channels, '\n')
        
        
        self.conv1 = md.Conv2dReLU(
            # 2 * (in_channels + skip_channels), # -------->> use for concatenation <<----------
            in_channels + skip_channels, # -------->> use for addition <<----------
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        
        self.conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
               
        # P-scSE
        if self.attention_type == 'pscse':            
            self.attention1 = md.SCSEModule0(in_channels=in_channels + skip_channels, strategy='maxout') 
            self.attention2 = md.SCSEModule0(in_channels=in_channels + skip_channels, strategy='addition')
        
        # scSE
        elif self.attention_type == 'scse':
            self.attention1 = md.Attention("scse", in_channels=in_channels + skip_channels)
            self.attention2 = md.Attention("scse", in_channels=out_channels)
        
        # Single attention (as per the original paper -> attention is applied at the end)
        elif self.attention_type in ('maxout', 'additive', 'concat', 'multiplication', 'average', 'all-average'):
            self.attention = md.SCSEModule0(in_channels=out_channels, strategy=self.attention_type)
            
        elif self.attention_type == None:
            pass
            # self.attention1 = md.Attention(attention_type, in_channels=in_channels + skip_channels)
            # self.attention2 = md.Attention(attention_type, in_channels=out_channels)
            
        else: 
            raise ValueError("Wrong keyword for attention. Choose either of - pscse, scse, \
                             maxout, additive, concat, multiplication, average, all-average")

    def forward(self, x, skip=None, strategy='add'):
               
        # For P-scSE
        if self.attention_type == 'pscse':
            # x = F.interpolate(x, scale_factor=2, mode="nearest")
            
            x_upsampled = F.interpolate(x, scale_factor=2, mode="nearest")
            
            if skip is not None:   

                if self.skip_channels > 0: # added for MiT
                            
                    x_gate = self.gate(x)
                    skip = self.gated_attention(g=x_gate, x=skip) # new attention skip

                x_cat = torch.cat([x_upsampled, skip], dim=1)
                x1 = self.attention1(x_cat) # 1st attention         
                x2 = self.attention2(x_cat) # 2nd attention          

                if strategy == 'cat': x3 = torch.cat([x1, x2], dim=1) # concatenate two attentions   
                elif strategy == 'add': x3 = x1 + x2
                
            else:
                x2 = self.attention2(x_upsampled) # 2nd attention
                if strategy == 'cat': x3 = torch.cat([x_upsampled, x2], dim=1) # concatenate x_upsampled and x2
                elif strategy == 'add': x3 = x_upsampled + x2
            
            x3 = self.conv1(x3)
            
            x3 = self.dropout(x3)

            return x3
        
        # For scSE
        elif self.attention_type == 'scse':
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            if skip is not None:
                x = torch.cat([x, skip], dim=1)
                x = self.attention1(x)
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.attention2(x)
            return x
        
        # No attention
        elif self.attention_type == None:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            if skip is not None:
                x = torch.cat([x, skip], dim=1)
                # x = self.attention1(x)
            x = self.conv1(x)
            x = self.conv2(x)
            # x = self.attention2(x)
            return x
        
        # Single attention (as per the original paper -> attention is applied at the end)
        else: 
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            if skip is not None:
                x = torch.cat([x, skip], dim=1)
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.attention(x)
            return x

# =========================================================================
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
