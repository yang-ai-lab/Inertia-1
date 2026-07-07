"""
PatchTST for 3D Accelerometer Data
Treats x, y, z channels as a single 3D vector instead of independent channels
"""

__all__ = ['PatchTST_Accel']

from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

from collections import OrderedDict
from .layers.pos_encoding import *
from .layers.basics import *
from .layers.attention import *


class PatchTST_Accel(nn.Module):
    """
    PatchTST adapted for 3-channel accelerometer data.
    Instead of processing each channel independently, treats x/y/z as a 3D vector.
    
    Output dimension: 
         [bs x num_patch x 3 x patch_len] for pretrain
         [bs x target_dim x 3] for prediction
         [bs x n_classes] for classification
    """
    def __init__(self, patch_len:int, stride:int, num_patch:int, 
                 n_layers:int=3, d_model=128, n_heads=16, d_ff:int=256, 
                 norm:str='LayerNorm', attn_dropout:float=0., dropout:float=0., act:str="gelu", 
                 res_attention:bool=True, pre_norm:bool=False, store_attn:bool=False,
                 pe:str='zeros', learn_pe:bool=True, head_dropout = 0, 
                 head_type = "pretrain", target_dim=None,
                 verbose:bool=False, **kwargs):

        super().__init__()

        assert head_type in ['pretrain', 'prediction', 'classification'], \
            'head type should be pretrain, prediction, or classification'
        
        # Backbone - processes 3D accelerometer patches
        self.backbone = AccelPatchTSTEncoder(
            num_patch=num_patch, 
            patch_len=patch_len, 
            n_layers=n_layers, 
            d_model=d_model, 
            n_heads=n_heads, 
            d_ff=d_ff,
            attn_dropout=attn_dropout, 
            dropout=dropout, 
            act=act, 
            res_attention=res_attention, 
            pre_norm=pre_norm, 
            store_attn=store_attn,
            pe=pe, 
            learn_pe=learn_pe, 
            verbose=verbose, 
            **kwargs
        )

        # Head
        self.head_type = head_type

        if head_type == "pretrain":
            # Reconstruct masked patches
            self.head = AccelPretrainHead(d_model, patch_len, head_dropout)
        elif head_type == "prediction":
            # Forecast future accelerometer values
            assert target_dim is not None, "target_dim required for prediction"
            self.head = AccelPredictionHead(d_model, num_patch, target_dim, head_dropout)
        elif head_type == "classification":
            # Activity classification
            assert target_dim is not None, "target_dim (num_classes) required for classification"
            self.head = AccelClassificationHead(d_model, num_patch, target_dim, head_dropout)


    def forward(self, x):                             
        """
        x: tensor [bs x num_patch x 3 x patch_len]
        """   
        z = self.backbone(x)  # z: [bs x num_patch x d_model]
        z = self.head(z)      
        # z: [bs x num_patch x 3 x patch_len] for pretrain
        #    [bs x target_dim x 3] for prediction
        #    [bs x n_classes] for classification
        return z


class AccelPretrainHead(nn.Module):
    """Reconstruct masked 3D accelerometer patches"""
    def __init__(self, d_model, patch_len, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # Output 3 channels (x, y, z) per timestep
        self.linear = nn.Linear(d_model, patch_len * 3)
        self.patch_len = patch_len

    def forward(self, x):
        """
        x: tensor [bs x num_patch x d_model]
        output: tensor [bs x num_patch x 3 x patch_len]
        """
        x = self.dropout(x)
        x = self.linear(x)  # [bs x num_patch x (patch_len * 3)]
        
        # Reshape to [bs x num_patch x 3 x patch_len]
        bs, num_patch = x.shape[0], x.shape[1]
        x = x.view(bs, num_patch, 3, self.patch_len)
        
        return x


class AccelPredictionHead(nn.Module):
    """Forecast future 3D accelerometer values"""
    def __init__(self, d_model, num_patch, forecast_len, dropout=0):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.dropout = nn.Dropout(dropout)
        # Predict forecast_len timesteps for 3 channels
        self.linear = nn.Linear(num_patch * d_model, forecast_len * 3)
        self.forecast_len = forecast_len

    def forward(self, x):                     
        """
        x: [bs x num_patch x d_model]
        output: [bs x forecast_len x 3]
        """
        x = self.flatten(x)  # [bs x (num_patch * d_model)]
        x = self.dropout(x)
        x = self.linear(x)   # [bs x (forecast_len * 3)]
        
        # Reshape to [bs x forecast_len x 3]
        bs = x.shape[0]
        x = x.view(bs, self.forecast_len, 3)
        
        return x


class AccelClassificationHead(nn.Module):
    """Activity classification from 3D accelerometer data"""
    def __init__(self, d_model, num_patch, n_classes, dropout=0):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool1d(1)  # Global average pooling
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes)
        )

    def forward(self, x):
        """
        x: [bs x num_patch x d_model]
        output: [bs x n_classes]
        """
        # Transpose for pooling
        x = x.transpose(1, 2)  # [bs x d_model x num_patch]
        x = self.gap(x)        # [bs x d_model x 1]
        x = x.squeeze(-1)      # [bs x d_model]
        x = self.dropout(x)
        x = self.fc(x)         # [bs x n_classes]
        return x


class AccelPatchTSTEncoder(nn.Module):
    """
    Encoder for 3D accelerometer data.
    Processes patches of [patch_len x 3] as 3D vectors.
    """
    def __init__(self, num_patch, patch_len, 
                 n_layers=3, d_model=128, n_heads=16,
                 d_ff=256, norm='LayerNorm', attn_dropout=0., dropout=0., 
                 act="gelu", store_attn=False,
                 res_attention=True, pre_norm=False,
                 pe='zeros', learn_pe=True, verbose=False, **kwargs):

        super().__init__()
        self.num_patch = num_patch
        self.patch_len = patch_len
        self.d_model = d_model

        # Input encoding: project [patch_len x 3] patches to d_model
        # Each patch is flattened: patch_len * 3 -> d_model
        self.W_P = nn.Linear(patch_len * 3, d_model)

        # Positional encoding for patches
        self.W_pos = positional_encoding(pe, learn_pe, num_patch, d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

        # Transformer encoder
        self.encoder = TSTEncoder(
            d_model, n_heads, d_ff=d_ff, norm=norm, 
            attn_dropout=attn_dropout, dropout=dropout,
            pre_norm=pre_norm, activation=act, 
            res_attention=res_attention, n_layers=n_layers, 
            store_attn=store_attn
        )

    def forward(self, x) -> Tensor:          
        """
        x: tensor [bs x num_patch x 3 x patch_len]
        output: tensor [bs x num_patch x d_model]
        """
        bs, num_patch, n_channels, patch_len = x.shape
        assert n_channels == 3, "Expected 3 accelerometer channels (x, y, z)"
        
        # Flatten each 3D patch: [bs x num_patch x 3 x patch_len] -> [bs x num_patch x (3*patch_len)]
        x = x.reshape(bs, num_patch, -1)
        
        # Project to d_model
        x = self.W_P(x)  # [bs x num_patch x d_model]
        
        # Add positional encoding
        x = self.dropout(x + self.W_pos)  # [bs x num_patch x d_model]

        # Transformer encoding
        z = self.encoder(x)  # [bs x num_patch x d_model]

        return z


# Cell
class TSTEncoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=None, 
                        norm='LayerNorm', attn_dropout=0., dropout=0., activation='gelu',
                        res_attention=False, n_layers=1, pre_norm=False, store_attn=False):
        super().__init__()

        self.layers = nn.ModuleList([TSTEncoderLayer(d_model, n_heads=n_heads, d_ff=d_ff, norm=norm,
                                                      attn_dropout=attn_dropout, dropout=dropout,
                                                      activation=activation, res_attention=res_attention,
                                                      pre_norm=pre_norm, store_attn=store_attn) for i in range(n_layers)])
        self.res_attention = res_attention
        self.pre_norm = pre_norm
        
        if self.pre_norm:
            if "batch" in norm.lower():
                self.norm = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
            else:
                self.norm = nn.LayerNorm(d_model)
        else:
            self.norm = None

    def forward(self, src:Tensor):
        """
        src: tensor [bs x num_patch x d_model]
        """
        output = src
        scores = None
        if self.res_attention:
            for mod in self.layers: output, scores = mod(output, prev=scores)
        else:
            for mod in self.layers: output = mod(output)
            
        if self.pre_norm and self.norm is not None:
            output = self.norm(output)
            
        return output


class TSTEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=256, store_attn=False,
                 norm='LayerNorm', attn_dropout=0, dropout=0., bias=True, 
                 activation="gelu", res_attention=False, pre_norm=False):
        super().__init__()
        assert not d_model%n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads
        d_v = d_model // n_heads

        # Multi-Head attention
        self.res_attention = res_attention
        self.self_attn = MultiheadAttention(d_model, n_heads, d_k, d_v, attn_dropout=attn_dropout, proj_dropout=dropout, res_attention=res_attention)

        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_attn = nn.LayerNorm(d_model)

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
                                get_activation_fn(activation),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model, bias=bias))

        # Add & Norm
        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_ffn = nn.LayerNorm(d_model)

        self.pre_norm = pre_norm
        self.store_attn = store_attn


    def forward(self, src:Tensor, prev:Optional[Tensor]=None):
        """
        src: tensor [bs x num_patch x d_model]
        """
        # Multi-Head attention sublayer
        if self.pre_norm:
            src2 = self.norm_attn(src)
        else:
            src2 = src

        ## Multi-Head attention
        if self.res_attention:
            src2, attn, scores = self.self_attn(src2, src2, src2, prev)
        else:
            src2, attn = self.self_attn(src2, src2, src2)
        
        if self.store_attn:
            self.attn = attn
            
        ## Add & Norm
        src = src + self.dropout_attn(src2)
        if not self.pre_norm:
            src = self.norm_attn(src)

        # Feed-forward sublayer
        if self.pre_norm:
            src2 = self.norm_ffn(src)
        else:
            src2 = src

        ## Position-wise Feed-Forward
        src2 = self.ff(src2)
        
        ## Add & Norm
        src = src + self.dropout_ffn(src2)
        if not self.pre_norm:
            src = self.norm_ffn(src)

        if self.res_attention:
            return src, scores
        else:
            return src
