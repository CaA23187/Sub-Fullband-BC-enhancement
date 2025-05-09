from operator import length_hint, mod
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.models as M
from torch import Tensor
import numpy as np
from causal_norm import CumulativeLayerNorm1d

### This is Causal/non-Causal version of new new sConformer. The old version modified Batch_first and added causal model, but neglecting the effect of Norm layer on causality


class _ConvolutionModule(torch.nn.Module):
    r"""Conformer convolution module.

    Args:
        input_dim (int): input dimension.
        num_channels (int): number of depthwise convolution layer input channels.
        depthwise_kernel_size (int): kernel size of depthwise convolution layer.
        bias (bool, optional): indicates whether to add bias term to each convolution layer. (Default: ``False``)
    """

    def __init__(
        self,
        input_dim: int,
        num_channels: int,
        depthwise_kernel_size: int,
        bias: bool = False,
        dropout: float = 0.0,
        causal: bool=False,
    ) -> None:
        super().__init__()
        # assert (depthwise_kernel_size - 1) % 2 == 0, "depthwise_kernel_size must be odd to achieve 'SAME' padding."
        if causal == True:
            self.layer_norm = CumulativeLayerNorm1d(input_dim) # cLN

            self.sequential = torch.nn.Sequential(
                torch.nn.Conv1d(
                    input_dim,
                    2 * num_channels,
                    1,
                    stride=1,
                    padding=0,
                    bias=bias,
                ),
                torch.nn.GLU(dim=1),
                torch.nn.ConstantPad1d((depthwise_kernel_size-1,0), 0.),
                torch.nn.Conv1d(
                    num_channels,
                    num_channels,
                    depthwise_kernel_size,
                    stride=1,
                    # padding=(depthwise_kernel_size - 1) // 2,
                    groups=num_channels,
                    bias=bias,
                ),
                torch.nn.BatchNorm1d(num_channels),
                torch.nn.SiLU(),
                torch.nn.Conv1d(
                    num_channels,
                    input_dim,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=bias,
                ),
                torch.nn.Dropout(dropout),
            )
        else:
            self.layer_norm = torch.nn.LayerNorm(input_dim)

            self.sequential = torch.nn.Sequential(
                torch.nn.Conv1d(
                    input_dim,
                    2 * num_channels,
                    1,
                    stride=1,
                    padding=0,
                    bias=bias,
                ),
                torch.nn.GLU(dim=1),
                torch.nn.Conv1d(
                    num_channels,
                    num_channels,
                    depthwise_kernel_size,
                    stride=1,
                    padding=(depthwise_kernel_size - 1) // 2,
                    groups=num_channels,
                    bias=bias,
                ),
                torch.nn.BatchNorm1d(num_channels),
                torch.nn.SiLU(),
                torch.nn.Conv1d(
                    num_channels,
                    input_dim,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=bias,
                ),
                torch.nn.Dropout(dropout),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            input (torch.Tensor): with shape `(B, T, D)`.

        Returns:
            torch.Tensor: output, with shape `(B, T, D)`.
        """
        x = self.layer_norm(x)
        # x = x.transpose(1, 2).contiguous()
        x = x.transpose(1, 2).to(memory_format=torch.contiguous_format)
        x = self.sequential(x)
        # return x.transpose(1, 2).contiguous()
        return x.transpose(1, 2).to(memory_format=torch.contiguous_format)


class _FeedForwardModule(torch.nn.Module):
    r"""Positionwise feed forward layer.

    Args:
        input_dim (int): input dimension.
        hidden_dim (int): hidden dimension.
        dropout (float, optional): dropout probability. (Default: 0.0)
    """

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.0, causal: bool=False,) -> None:
        super().__init__()
        if causal:
            self.sequential = torch.nn.Sequential(
                CumulativeLayerNorm1d(input_dim),
                torch.nn.Linear(input_dim, hidden_dim, bias=True),
                torch.nn.SiLU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(hidden_dim, input_dim, bias=True),
                torch.nn.Dropout(dropout),
            )
        else:
            self.sequential = torch.nn.Sequential(
                torch.nn.LayerNorm(input_dim),
                torch.nn.Linear(input_dim, hidden_dim, bias=True),
                torch.nn.SiLU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(hidden_dim, input_dim, bias=True),
                torch.nn.Dropout(dropout),
            )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            input (torch.Tensor): with shape `(*, D)`.

        Returns:
            torch.Tensor: output, with shape `(*, D)`.
        """
        return self.sequential(input)


class CausalSelfAttention(nn.Module):
    def __init__(self, num_heads: int, embed_dimension: int, bias: bool=False, is_causal: bool=False, dropout:float=0.0, lookahead=0):
        super().__init__()
        assert embed_dimension % num_heads == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(embed_dimension, 3 * embed_dimension, bias=bias)
        # output projection
        self.c_proj = nn.Linear(embed_dimension, embed_dimension, bias=bias)
        # regularization
        self.dropout = dropout
        self.resid_dropout = nn.Dropout(dropout)
        self.num_heads = num_heads
        self.embed_dimension = embed_dimension
        # Perform causal masking
        self.is_causal = is_causal

        self.lookahead = lookahead 

    def forward(self, x):
        '''
        x: (B, T, D)
        '''
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        query_projected = self.c_attn(x)

        batch_size = query_projected.size(0)
        embed_dim = query_projected.size(2)
        head_dim = embed_dim // (self.num_heads * 3)

        query, key, value = query_projected.chunk(3, -1)
        query = query.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2).contiguous()
        key = key.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2).contiguous()
        value = value.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2).contiguous()

        if self.training:
            dropout = self.dropout
        else:
            dropout = 0.0
        
        if self.is_causal:
            attn_mask = torch.ones(query.shape[2], query.shape[2], dtype=torch.bool, device=query.device).tril(diagonal=self.lookahead) 
            y = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, dropout_p=dropout)
        else:
            y = F.scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=dropout, is_causal=False)
        y = y.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * head_dim)

        y = self.resid_dropout(self.c_proj(y))
        return y
        


class sConformerLayer(torch.nn.Module):
    r"""sConformer layer that constitutes Conformer.

    Args:
        input_dim (int): input dimension.
        ffn_dim (int): hidden layer dimension of feedforward network.
        num_attention_heads (int): number of attention heads.
        depthwise_conv_kernel_size (int): kernel size of depthwise convolution layer.
        dropout (float, optional): dropout probability. (Default: 0.0)
    """

    def __init__(
        self,
        input_dim: int,
        ffn_dim: int,
        num_attention_heads: int,
        depthwise_conv_kernel_size: int,
        dropout: float = 0.0,
        causal: bool = False,
        lookahead: int = 0,
    ) -> None:
        super().__init__()

        # Conv Model
        self.conv_module = _ConvolutionModule(
            input_dim=input_dim,
            num_channels=input_dim,
            depthwise_kernel_size=depthwise_conv_kernel_size,
            bias=True,
            dropout=dropout,
            causal=causal,
        )

        # Selt-Attention
        if causal == True:
            self.self_attn_layer_norm = CumulativeLayerNorm1d(input_dim) # cLN
        else:
            self.self_attn_layer_norm = torch.nn.LayerNorm(input_dim)
        
        self.self_attn = CausalSelfAttention(embed_dimension=input_dim, num_heads=num_attention_heads, dropout=dropout, is_causal=causal, bias=True, lookahead=lookahead)
        self.self_attn_dropout = torch.nn.Dropout(dropout)

        # FFN
        self.ffn1 = _FeedForwardModule(input_dim, ffn_dim, dropout=dropout,causal=causal)
        if causal == True:
            self.final_layer_norm = CumulativeLayerNorm1d(input_dim)
        else:
            self.final_layer_norm = torch.nn.LayerNorm(input_dim)

    def forward(self, input: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None, attn_mask: Optional[Tensor] = None) -> torch.Tensor:
        r"""
        Args:
            input (torch.Tensor): input, with shape `(B, T, D)`.
            key_padding_mask (torch.Tensor or None): key padding mask to use in self attention layer.
            attn_mask (torch.Tensor or None): If specified, a 2D or 3D mask preventing attention to certain positions.For a binary mask, a ``True`` value indicates that the
                corresponding position is not allowed to attend.
        Returns:
            torch.Tensor: output, with shape `(B, T, D)`.
        """
        
        residual = input
        x = input
        x = self.conv_module(x)
        x = residual + x
        
        residual = x
        x = self.self_attn_layer_norm(x)
        x = self.self_attn(x)
        x = self.self_attn_dropout(x)
        x = x + residual

        residual = x
        x = self.ffn1(x)
        x = x + residual

        x = self.final_layer_norm(x)
        return x


class sConformer(torch.nn.Module):
    r"""Implements the Conformer architecture introduced in
    *Conformer: Convolution-augmented Transformer for Speech Recognition*
    [:footcite:`gulati2020conformer`].

    Args:
        input_dim (int): input dimension.
        num_heads (int): number of attention heads in each Conformer layer.
        ffn_dim (int): hidden layer dimension of feedforward networks.
        num_layers (int): number of Conformer layers to instantiate.
        depthwise_conv_kernel_size (int): kernel size of each Conformer layer's depthwise convolution layer.
        dropout (float, optional): dropout probability. (Default: 0.0)

    Examples:
        >>> conformer = Conformer(
        >>>     input_dim=80,
        >>>     num_heads=4,
        >>>     ffn_dim=128,
        >>>     num_layers=4,
        >>>     depthwise_conv_kernel_size=31,
        >>> )
        >>> lengths = torch.randint(1, 400, (10,))  # (batch,)
        >>> input = torch.rand(10, int(lengths.max()), input_dim)  # (batch, num_frames, input_dim)
        >>> output = conformer(input, lengths)
    """

    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        ffn_dim: int,
        num_layers: int,
        depthwise_conv_kernel_size: int,
        dropout: float = 0.0,
        causal: bool=False,
        lookahead: int=0,
    ):
        super().__init__()
        self.causal = causal

        if causal:
            self.conformer_layers = torch.nn.ModuleList(
                [
                    sConformerLayer(
                        input_dim,
                        ffn_dim,
                        num_heads,
                        depthwise_conv_kernel_size,
                        dropout,
                        causal=causal,
                        lookahead=lookahead,
                    )
                    for _ in range(num_layers)
                ]
            )
        else:
            self.conformer_layers = torch.nn.ModuleList(
                [
                    sConformerLayer(
                        input_dim,
                        ffn_dim,
                        num_heads,
                        depthwise_conv_kernel_size,
                        dropout,
                        causal=causal,
                    )
                    for _ in range(num_layers)
                ]
            )

    def forward(self, input: torch.Tensor, lengths=None) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Args:
            input (torch.Tensor): with shape `(B, T, input_dim)`.
            lengths (torch.Tensor): with shape `(B,)` and i-th element representing
                number of valid frames for i-th batch element in ``input``.

        Returns:
            (torch.Tensor, torch.Tensor)
                torch.Tensor
                    output frames, with shape `(B, T, input_dim)`
                torch.Tensor
                    output lengths, with shape `(B,)` and i-th element representing
                    number of valid frames for i-th batch element in output frames.
        """
    
        for layer in self.conformer_layers:
            input = layer(input)
        return input, None


class sConformerLayer_testMACS(torch.nn.Module):
    r"""sConformer layer that constitutes Conformer.

    Args:
        input_dim (int): input dimension.
        ffn_dim (int): hidden layer dimension of feedforward network.
        num_attention_heads (int): number of attention heads.
        depthwise_conv_kernel_size (int): kernel size of depthwise convolution layer.
        dropout (float, optional): dropout probability. (Default: 0.0)
    """

    def __init__(
        self,
        input_dim: int,
        ffn_dim: int,
        num_attention_heads: int,
        depthwise_conv_kernel_size: int,
        dropout: float = 0.0,
        causal: bool = False,
        lookahead: int = 0,
    ) -> None:
        super().__init__()

        # Conv Model
        self.conv_module = _ConvolutionModule(
            input_dim=input_dim,
            num_channels=input_dim,
            depthwise_kernel_size=depthwise_conv_kernel_size,
            bias=True,
            dropout=dropout,
            causal=causal,
        )

        # Selt-Attention
        if causal == True:
            self.self_attn_layer_norm = CumulativeLayerNorm1d(input_dim) # cLN
        else:
            self.self_attn_layer_norm = torch.nn.LayerNorm(input_dim)
        
        self.self_attn = CausalSelfAttention(embed_dimension=input_dim, num_heads=num_attention_heads, dropout=dropout, is_causal=causal, bias=True, lookahead=lookahead)
        self.self_attn_dropout = torch.nn.Dropout(dropout)

        # FFN
        self.ffn1 = _FeedForwardModule(input_dim, ffn_dim, dropout=dropout,causal=causal)
        if causal == True:
            self.final_layer_norm = CumulativeLayerNorm1d(input_dim)
        else:
            self.final_layer_norm = torch.nn.LayerNorm(input_dim)

    def forward(self, input: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None, attn_mask: Optional[Tensor] = None) -> torch.Tensor:
        r"""
        Args:
            input (torch.Tensor): input, with shape `(B, T, D)`.
            key_padding_mask (torch.Tensor or None): key padding mask to use in self attention layer.
            attn_mask (torch.Tensor or None): If specified, a 2D or 3D mask preventing attention to certain positions.For a binary mask, a ``True`` value indicates that the
                corresponding position is not allowed to attend.
        Returns:
            torch.Tensor: output, with shape `(B, T, D)`.
        """
        
        residual = input
        x = input
        x = self.conv_module(x)
        x = residual + x
        
        residual = x
        x = self.self_attn_layer_norm(x)
        # x = self.self_attn(x)
        x = self.self_attn_dropout(x)
        x = x + residual

        residual = x
        x = self.ffn1(x)
        x = x + residual

        x = self.final_layer_norm(x)
        return x


class sConformer_testMACS(torch.nn.Module):
    r"""Implements the Conformer architecture introduced in
    *Conformer: Convolution-augmented Transformer for Speech Recognition*
    [:footcite:`gulati2020conformer`].

    Args:
        input_dim (int): input dimension.
        num_heads (int): number of attention heads in each Conformer layer.
        ffn_dim (int): hidden layer dimension of feedforward networks.
        num_layers (int): number of Conformer layers to instantiate.
        depthwise_conv_kernel_size (int): kernel size of each Conformer layer's depthwise convolution layer.
        dropout (float, optional): dropout probability. (Default: 0.0)

    Examples:
        >>> conformer = Conformer(
        >>>     input_dim=80,
        >>>     num_heads=4,
        >>>     ffn_dim=128,
        >>>     num_layers=4,
        >>>     depthwise_conv_kernel_size=31,
        >>> )
        >>> lengths = torch.randint(1, 400, (10,))  # (batch,)
        >>> input = torch.rand(10, int(lengths.max()), input_dim)  # (batch, num_frames, input_dim)
        >>> output = conformer(input, lengths)
    """

    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        ffn_dim: int,
        num_layers: int,
        depthwise_conv_kernel_size: int,
        dropout: float = 0.0,
        causal: bool=False,
        lookahead: int=0,
    ):
        super().__init__()
        self.causal = causal

        if causal:
            self.conformer_layers = torch.nn.ModuleList(
                [
                    sConformerLayer_testMACS(
                        input_dim,
                        ffn_dim,
                        num_heads,
                        depthwise_conv_kernel_size,
                        dropout,
                        causal=causal,
                        lookahead=lookahead,
                    )
                    for _ in range(num_layers)
                ]
            )
        else:
            self.conformer_layers = torch.nn.ModuleList(
                [
                    sConformerLayer_testMACS(
                        input_dim,
                        ffn_dim,
                        num_heads,
                        depthwise_conv_kernel_size,
                        dropout,
                        causal=causal,
                    )
                    for _ in range(num_layers)
                ]
            )

    def forward(self, input: torch.Tensor, lengths=None) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Args:
            input (torch.Tensor): with shape `(B, T, input_dim)`.
            lengths (torch.Tensor): with shape `(B,)` and i-th element representing
                number of valid frames for i-th batch element in ``input``.

        Returns:
            (torch.Tensor, torch.Tensor)
                torch.Tensor
                    output frames, with shape `(B, T, input_dim)`
                torch.Tensor
                    output lengths, with shape `(B,)` and i-th element representing
                    number of valid frames for i-th batch element in output frames.
        """
    
        for layer in self.conformer_layers:
            input = layer(input)
        return input, None


if __name__ == "__main__":
    data = torch.randn([4, 100, 80])
    model = sConformer_testMACS(input_dim=80,num_heads=4,ffn_dim=128,num_layers=4,depthwise_conv_kernel_size=31,dropout=0.1, causal=True)
    out, _ = model(data, lengths=torch.ones(4)*100)
    print(out.shape)

    from ptflops import get_model_complexity_info
    macs, params = get_model_complexity_info(model, (100, 80), as_strings=True,
                                           print_per_layer_stat=False, verbose=False, output_precision=5)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    print(f'{model._get_name()}, Real MACs: ', float(macs.split(' ')[0])/4)

    # num_heads = 8
    # heads_per_dim = 64
    # embed_dimension = num_heads * heads_per_dim
    # dtype = torch.float16
    # model = CausalSelfAttention(num_heads=num_heads, embed_dimension=embed_dimension, bias=False, is_causal=True, dropout=0.1)#.to("cuda").to(dtype).eval()
    # print(model)

    # B,T,dim = 4, 100, 8*64
    # data = torch.randn([B,T,dim]).to("cuda").to(dtype)
    # out = model(data)
    # print("out: ", out.shape)
    
    ## causal mask
    # attn_mask = torch.ones(10, 10, dtype=torch.bool).tril(diagonal=1) 
    # print(attn_mask)
