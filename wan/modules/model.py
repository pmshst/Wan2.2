# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.

# optimized to run inference on 8GB VRAM
import gc
import math
import torch
import time
import logging
import torch.nn as nn
import safetensors.torch

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

from .attention import flash_attention, flash_attention_single

from torch.nn import functional as F, init
from torch import Tensor

logging.basicConfig(level=logging.DEBUG)

#from line_profiler import profile
#from memory_profiler import profile

# for 3070 torch.float16 faster
# use torch.bfloat16 for cards with native support A40, A6000, RTX 3080
dtype_c = torch.float16


def print_gpu_memory_report():
    print("--- PyTorch CUDA Memory Summary ---")
    print(torch.cuda.memory_summary())

    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)
    print(f"Current Memory (Allocated by Tensors): {allocated:.2f} GB")
    print(f"Cached Memory (Reserved by PyTorch):   {reserved:.2f} GB")
    print("-" * 40)
    print("\n")

    print("--- Detailed CUDA Tensor Breakdown ---")
    all_objects = gc.get_objects()
    cuda_tensors = [
        obj for obj in all_objects
        if torch.is_tensor(obj) and obj.is_cuda
    ]

    if not cuda_tensors:
        print("No CUDA tensors found in memory by the garbage collector.")
        return

    table_data = []
    total_size_mb = 0
    for tensor in cuda_tensors:
        size_bytes = tensor.nelement() * tensor.element_size()
        size_mb = size_bytes / (1024 * 1024)
        total_size_mb += size_mb

        table_data.append({
            "id": id(tensor),
            "device": tensor.device,
            "size_mb": size_mb,
            "dtype": tensor.dtype,
            "shape": tuple(tensor.shape),
        })

    table_data.sort(key=lambda x: x['size_mb'], reverse=True)

    header = f"{'ID':>15} | {'Device':<10} | {'Size (MB)':>12} | {'Dtype':<18} | {'Shape'}"
    print(header)
    print("-" * 100)
    for item in table_data:
        row = (
            f"{item['id']:>15} | "
            f"{str(item['device']):<10} | "
            f"{item['size_mb']:>12.2f} | "
            f"{str(item['dtype']):<18} | "
            f"{item['shape']}"
        )
        print(row)

    print("-" * 100)
    print(f"Total Tensors Found: {len(table_data)}")
    print(f"Total Size of Tensors (from gc): {total_size_mb / 1024:.2f} GB ({total_size_mb:.2f} MB)")
    print("-" * 100)


class nnLinear2(nn.Module):
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=dtype_c,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.weight, self.bias)


class nnLinear(nn.Module):
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=dtype_c,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()
        self.cache = {}

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    #@profile
    def forward(self, input: Tensor, source="", i=None) -> Tensor:
        if self.cache:
            x = F.linear(input, self.cache['w'], self.cache['b'])
            self.cache = {}
            return x
        self.cache['w'] = self.weight
        self.cache['b'] = self.bias
        return F.linear(input, self.cache['w'], self.cache['b'])
        #print(source)
        #print(self.weight.size())
        if source == "qkv_fn q":
            return F.linear(input, self.weight, self.bias)
        if source == "qkv_fn k":
            return F.linear(input, self.weight, self.bias)
        if source == "qkv_fn v":
            return F.linear(input, self.weight, self.bias)
        if source == "qkv_fn o":
            return F.linear(input, self.weight, self.bias)
        if source == "q cross":
            return F.linear(input, self.weight, self.bias)
        if source == "k cross":
            return F.linear(input, self.weight, self.bias)
        if source == "v cross":
            return F.linear(input, self.weight, self.bias)
        if source == "o cross":
            return F.linear(input, self.weight, self.bias)

        #return F.linear(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"


class DynamicSwapInstaller:
    @staticmethod
    def _install_module(module: torch.nn.Module, **kwargs):
        original_class = module.__class__
        module.__dict__['forge_backup_original_class'] = original_class

        def hacked_get_attr(self, name: str):
            if '_parameters' in self.__dict__:
                _parameters = self.__dict__['_parameters']
                if name in _parameters:
                    p = _parameters[name]
                    if p is None:
                        return None
                    if p.__class__ == torch.nn.Parameter:
                        return p.to(device="cuda")
                    else:
                        n1 = p.to(device="cuda")
                        return n1.to(dtype=dtype_c)
            if '_buffers' in self.__dict__:
                _buffers = self.__dict__['_buffers']
                if name in _buffers:
                    n1 = _buffers[name].to(device="cuda")
                    return n1.to(dtype=dtype_c)
            return super(original_class, self).__getattr__(name)

        module.__class__ = type('DynamicSwap_' + original_class.__name__, (original_class,), {
            '__getattr__': hacked_get_attr,
        })
        module.n = {}

        return

    @staticmethod
    def install_model(model: torch.nn.Module, **kwargs):
        count_modules = 0
        for m in model.modules():
            count_modules += 1
            DynamicSwapInstaller._install_module(m, **kwargs)
        return


class DynamicSwapInstaller2:
    @staticmethod
    def _install_module(module: torch.nn.Module, **kwargs):
        original_class = module.__class__
        module.__dict__['forge_backup_original_class'] = original_class

        def hacked_get_attr(self, name: str):
            if '_parameters' in self.__dict__:
                _parameters = self.__dict__['_parameters']
                if name in _parameters:
                    p = _parameters[name]
                    if p is None:
                        return None
                    if p.__class__ == torch.nn.Parameter:
                        return p.to(device="cuda").to(dtype=dtype_c)
                    else:
                        n1 = p.to(device="cuda")
                        return n1.to(dtype=dtype_c)
            if '_buffers' in self.__dict__:
                _buffers = self.__dict__['_buffers']
                if name in _buffers:
                    return _buffers[name]
                    n1 = _buffers[name].to(device="cuda")
                    return n1.to(dtype=dtype_c)
            return super(original_class, self).__getattr__(name)

        module.__class__ = type('DynamicSwap_' + original_class.__name__, (original_class,), {
            '__getattr__': hacked_get_attr,
        })
        module.n = {}

        return

    @staticmethod
    def install_model(model: torch.nn.Module, **kwargs):
        count_modules = 0
        for m in model.modules():
            count_modules += 1
            DynamicSwapInstaller2._install_module(m, **kwargs)
        return


class DynamicQuantizer:
    @staticmethod
    def _install_module(module: torch.nn.Module, device):
        original_class = module.__class__

        module.__dict__['forge_backup_original_class'] = original_class
        def hacked_get_attr(self, name: str):
            if '_parameters' in self.__dict__:
                _parameters = self.__dict__['_parameters']
                if name in _parameters:
                    p = _parameters[name]
                    if p is None:
                        return None
                    if p.__class__ == torch.nn.Parameter:
                        if p.device != device:
                            p = p.to(device)
                        return p.to(dtype=dtype_c)
                    else:
                        if p.device != device:
                            p = p.to(device)
                        return p.to(dtype=dtype_c)
            if '_buffers' in self.__dict__:
                _buffers = self.__dict__['_buffers']
                if name in _buffers:
                    n1 = _buffers[name].to(device="cuda")
                    return n1.to(dtype=dtype_c)
            return super(original_class, self).__getattr__(name)

        def hacked_set_attr(self, name, value):
            if '_parameters' in self.__dict__:
                _parameters = self.__dict__['_parameters']
                if value.__class__ == torch.nn.Parameter:
                    value = value.to(dtype=torch.float8_e5m2)
                    value = torch.nn.Parameter(value, requires_grad=False)
                    return super(original_class, self).__setattr__(name, value)

            return super(original_class, self).__setattr__(name, value)

        module.__class__ = type('DynamicSwap_' + original_class.__name__, (original_class,), {
            '__getattr__': hacked_get_attr,
            '__setattr__': hacked_set_attr,
        })

        return

    @staticmethod
    def install_model(model: torch.nn.Module, device):
        count_modules = 0
        for m in model.modules():
            count_modules += 1
            DynamicQuantizer._install_module(m, device)
        return


__all__ = ['WanModel']


def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta,
                        torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


#@torch.amp.autocast('cuda', enabled=False)
def rope_apply_(x, grid_sizes, freqs):
    global g_freqs

    n, c = x.size(2), x.size(3) // 2
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float32).reshape(seq_len, n, -1, 2))
        freqs_i = torch.cat([
            g_freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            g_freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            g_freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i.to(dtype_c))
    return torch.stack(output)


def rope_apply(x, grid_sizes):
    global freqs_i
    n, c = x.size(2), x.size(3) // 2
    f, h, w = grid_sizes.tolist()[0]
    seq_len = f * h * w
    x_complex = torch.view_as_complex(x[0, :seq_len].to(torch.float32).reshape(seq_len, n, c, 2))
    x_real = torch.view_as_real(x_complex * freqs_i).flatten(2).to(dtype_c)
    x[0, :seq_len] = x_real
    return x


class WanRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = torch.tensor(eps, dtype=dtype_c, device="cuda")
        self.weight = nn.Parameter(torch.ones(dim, dtype=dtype_c))
        self.cache = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.cache:
            x = x * torch.rsqrt(
                x.pow(2).mean(-1, keepdim=True, dtype=dtype_c) + self.eps,
            ) * self.cache['w']
            self.cache = {}
            return x

        self.cache['w'] = self.weight
        return x * torch.rsqrt(
            x.pow(2).mean(-1, keepdim=True, dtype=dtype_c) + self.eps,
        ) * self.cache['w']


class WanLayerNorm(nn.LayerNorm):

    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return super().forward(x)


from typing import List, Optional, Tuple, Union
_shape_t = Union[int, List[int]]
import numbers
class nnLayerNorm(nn.Module):
    __constants__ = ["normalized_shape", "eps", "elementwise_affine"]
    normalized_shape: Tuple[int, ...]
    eps: float
    elementwise_affine: bool

    def __init__(
        self,
        normalized_shape: _shape_t,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            # mypy error: incompatible types in assignment
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(
                torch.empty(self.normalized_shape, **factory_kwargs)
            )
            if bias:
                self.bias = nn.Parameter(
                    torch.empty(self.normalized_shape, **factory_kwargs)
                )
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self.reset_parameters()
        self.cache = {}

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            init.ones_(self.weight)
            if self.bias is not None:
                init.zeros_(self.bias)

    def forward(self, input: Tensor) -> Tensor:
        if self.cache:
            x = F.layer_norm(
                input, self.normalized_shape, self.cache['w'], self.cache['b'], self.cache['e']
            )
            self.cache = {}
            return x

        self.cache['w'] = self.weight
        self.cache['b'] = self.bias
        self.cache['e'] = self.eps
        x = F.layer_norm(
            input, self.normalized_shape, self.cache['w'], self.cache['b'], self.cache['e']
        )

        return x

    def extra_repr(self) -> str:
        return (
            "{normalized_shape}, eps={eps}, "
            "elementwise_affine={elementwise_affine}".format(**self.__dict__)
        )


class WanLayerNormD(nnLayerNorm):

    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return super().forward(x)


class WanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nnLinear(dim, dim, dtype=dtype_c)
        self.k = nnLinear(dim, dim, dtype=dtype_c)
        self.v = nnLinear(dim, dim, dtype=dtype_c)
        self.o = nnLinear(dim, dim, dtype=dtype_c)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    #@profile
    def forward(self, x, seq_lens, grid_sizes, i):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x, "qkv_fn q")).view(b, s, n, d)
            k = self.norm_k(self.k(x, "qkv_fn k")).view(b, s, n, d)
            v = self.v(x, "qkv_fn v").view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)
        q = rope_apply(q, grid_sizes)
        k = rope_apply(k, grid_sizes)

        #x = flash_attention(
        #    q=q,
        #    k=k,
        #    v=v,
        #    k_lens=seq_lens,
        #    window_size=self.window_size)
        #print(x)


        x = flash_attention_single(
            q=q[0],
            k=k[0],
            v=v[0],
            k_lens=seq_lens,
            window_size=self.window_size)
        x.unsqueeze(0)

        # output
        x = x.flatten(2)
        def o_func(x):
            return self.o(x, "qkv_fn o", i)

        x = o_func(x)
        return x


class WanCrossAttention(WanSelfAttention):

    def forward(self, x, context, context_lens, i):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x, "q cross")).view(b, -1, n, d)
        k = self.norm_k(self.k(context, "k cross")).view(b, -1, n, d)
        v = self.v(context, "v cross").view(b, -1, n, d)

        # compute attention
        x = flash_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        x = self.o(x, "o cross")
        return x


class WanAttentionBlock(nn.Module):

    def __init__(self,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNormD(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm,
                                          eps)
        self.norm3 = WanLayerNormD(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WanCrossAttention(dim, num_heads, (-1, -1), qk_norm,
                                            eps)
        self.norm2 = WanLayerNormD(dim, eps)
        self.ffn = nn.Sequential(
            nnLinear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nnLinear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.cache = {}

    def forward(
        self,
        x,
        x_null,
        e,
        seq_lens,
        grid_sizes,
        context,
        context_null,
        context_lens,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, L1, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        e = (self.modulation.unsqueeze(0) + e).chunk(6, dim=2)

        # self-attention
        x = x + self.self_attn(self.norm1(x) * (1 + e[1].squeeze(2)) + e[0].squeeze(2), seq_lens, grid_sizes) * e[2].squeeze(2)
        x_null = x_null + self.self_attn(self.norm1(x_null) * (1 + e[1].squeeze(2)) + e[0].squeeze(2), seq_lens, grid_sizes) * e[2].squeeze(2)

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens):
            x = x + self.cross_attn(self.norm3(x), context, context_lens)
            x = x + self.ffn(self.norm2(x) * (1 + e[4].squeeze(2)) + e[3].squeeze(2)) * e[5].squeeze(2)
            return x

        return cross_attn_ffn(x, context, context_lens), cross_attn_ffn(x_null, context_null, context_lens)

    #@profile
    def do_cross_attn(self, x, x_null, e, context, context_null, context_lens, i):
        if self.cache:
            modulation = self.cache['modulation']
            self.cache = {}
        else:
            self.cache['modulation'] = self.modulation
            modulation = self.cache['modulation']
        e_cf_combined = modulation.unsqueeze(0)[:, :, 3:, :] + e.to('cuda')#[:, :, 3:, :]
        e_cf = e_cf_combined.chunk(3, dim=2)

        x = x + self.cross_attn(self.norm3(x), context, context_lens, i)
        x = x + self.ffn(self.norm2(x) * (1 + e_cf[1].squeeze(2)) + e_cf[0].squeeze(2)) * e_cf[2].squeeze(2)

        x_null = x_null + self.cross_attn(self.norm3(x_null), context_null, context_lens, i)
        x_null = x_null + self.ffn(self.norm2(x_null) * (1 + e_cf[1].squeeze(2)) + e_cf[0].squeeze(2)) * e_cf[2].squeeze(2)

        return x, x_null

    #@profile
    def do_attn(self, x, x_null, e, seq_lens, grid_sizes, i):
        if self.cache:
            modulation = self.cache['modulation']
            self.cache = {}
        else:
            self.cache['modulation'] = self.modulation
            modulation = self.cache['modulation']
        e_sa_combined = modulation.unsqueeze(0)[:, :, :3, :] + e.to('cuda')#[:, :, :3, :]
        e_sa = e_sa_combined.chunk(3, dim=2)

        x = x + self.self_attn(self.norm1(x) * (1 + e_sa[1].squeeze(2)) + e_sa[0].squeeze(2), seq_lens, grid_sizes, i) * e_sa[2].squeeze(2)
        x_null = x_null + self.self_attn(self.norm1(x_null) * (1 + e_sa[1].squeeze(2)) + e_sa[0].squeeze(2), seq_lens, grid_sizes, i) * e_sa[2].squeeze(2)

        return x, x_null


class Head(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nnLinear2(dim, out_dim, dtype=dtype_c)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim, dtype=dtype_c) / dim**0.5)
        self.mod_0 = None
        self.mod_1 = None

    def set_modulation(self):
        mod_0, mod_1 = self.modulation.chunk(2, dim=1)
        self.mod_0 = mod_0.squeeze(1).to("cuda", dtype=dtype_c)
        self.mod_1 = mod_1.squeeze(1).to("cuda", dtype=dtype_c)

    def forward(self, x, e):
        if self.mod_0 is None:
            self.set_modulation()

        return self.head(self.norm(x) * (1 + self.mod_1 + e) + self.mod_0 + e)


freqs_i = None
blocks = None


class WanModel(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim', 'window_size'
    ]
    _no_split_modules = ['WanAttentionBlock']

    def clear_mem(self):
        self.blocks = None
        for i in range(self.num_layers):
            if getattr(self, f"blocks{i}"):
                delattr(self, f"blocks{i}")

        self.patch_embedding = None
        self.text_embedding = None
        self.time_embedding = None
        self.time_projection = None
        self.head = None
        gc.collect()
        torch.cuda.empty_cache()

    #@profile
    def _load_and_move_part_v1(self, part_name: str, part_class, *args, **kwargs):
        file_path = self.model_path + part_name + '.safetensors'
        attr_name = part_name.replace(".", "")

        if getattr(self, attr_name):
            return

        part_state_dict = safetensors.torch.load_file(file_path, device="cpu")

        if self.blocks_in_vram > 0 and part_name.split(".")[0] == "blocks" \
                and int(part_name.split(".")[1]) < self.blocks_in_vram:
            # loading to vram
            if getattr(self, attr_name) == None:
                with torch.device("meta"):
                    layer = part_class(*args, **kwargs)
                layer = layer.to_empty(device="cpu")

                layer.load_state_dict(part_state_dict, assign=True)
                layer.to('cuda')
                setattr(self, attr_name, layer)
                if self.load_as_fp8:
                    DynamicSwapInstaller2.install_model(getattr(self, attr_name), device="cuda")
            return

        if part_name.split(".")[0] == "blocks" \
                and int(part_name.split(".")[1]) >= self.num_layers - self.blocks_in_ram:
            # loading to ram
            with torch.device("meta"):
                layer = part_class(*args, **kwargs)
            layer = layer.to_empty(device="cpu")

            layer.load_state_dict(part_state_dict, assign=True)
            setattr(self, attr_name, layer)
            if self.load_as_fp8:
                DynamicSwapInstaller2.install_model(getattr(self, attr_name), device="cuda")
            else:
                DynamicSwapInstaller.install_model(getattr(self, attr_name), device="cuda")
            return

        if part_name.split(".")[0] == "blocks":
            attr_name = "blocks"
        if getattr(self, attr_name):
            # replacing block in vram from disc
            getattr(self, attr_name).load_state_dict(part_state_dict, assign=True)
        else:
            with torch.device("meta"):
                layer = part_class(*args, **kwargs)
            layer = layer.to_empty(device="cpu")
            layer.load_state_dict(part_state_dict, assign=True)
            #if part_name == "head":
            #    layer.to("cuda")

            setattr(self, attr_name, layer)
            #layer.to('cuda').to(dtype=dtype_c)
            #if part_name != "head":
            DynamicSwapInstaller2.install_model(getattr(self, attr_name), device="cuda")

        return

    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6,
                 blocks_in_ram=0,
                 blocks_in_vram=0,
                 load_as_fp8=False,
                 offload_large_tensors=False):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            window_size (`tuple`, *optional*, defaults to (-1, -1)):
                Window size for local attention (-1 indicates global attention)
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()
        self.load_as_fp8 = load_as_fp8
        self.offload_large_tensors = offload_large_tensors
        self.in_dim = in_dim
        if model_type == 'i2v_h' or model_type == 'i2v_l':
            self.in_dim = 36

        if model_type == 't2v_h':
            # todo: replace with your path to converted models see: optimize_files.py
            if load_as_fp8:
                self.model_path = "./Wan2.2-T2V-A14B/high_noise_model_fp8/"
            else:
                self.model_path = "./Wan2.2-T2V-A14B/high_noise_model/"

        if model_type == 't2v_l':
            # todo: replace with your path to converted models see: optimize_files.py
            if load_as_fp8:
                self.model_path = "./Wan2.2-T2V-A14B/low_noise_model_fp8/"
            else:
                self.model_path = "./Wan2.2-T2V-A14B/low_noise_model/"

        if model_type == 'i2v_h':
            # todo: replace with your path to converted models see: optimize_files.py
            if load_as_fp8:
                self.model_path = "./Wan2.2-I2V-A14B/high_noise_model_fp8/"
            else:
                self.model_path = "./Wan2.2-I2V-A14B/high_noise_model/"
        if model_type == 'i2v_l':
            # todo: replace with your path to converted models see: optimize_files.py
            if load_as_fp8:
                self.model_path = "./Wan2.2-I2V-A14B/low_noise_model_fp8/"
            else:
                self.model_path = "./Wan2.2-I2V-A14B/low_noise_model/"

        if model_type == 'ti2v':
            self.model_path = "./Wan2.2-TI2V-5B/model/"

        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # embeddings
        self.patch_embedding = None
        self.text_embedding = None

        self.time_embedding = None
        self.time_projection = None

        # blocks
        self.blocks = None
        for i in range(self.num_layers):
            setattr(self, f'blocks{i}', None)

        self.head = None

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ], dim=1)

        c = 64
        self.g_freqs = self.freqs.to("cuda").split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

        self.blocks_in_ram = blocks_in_ram
        logging.debug(f"\nRAM available for {self.blocks_in_ram}/{self.num_layers} blocks\n")

        self.blocks_in_vram = blocks_in_vram
        logging.debug(f"\nVRAM available for {self.blocks_in_vram}/{self.num_layers} blocks\n")

    #@profile
    def forward(
            self,
            x,
            t,
            context,
            context_null,
            seq_len,
            y=None,
    ):
        x = [el.to(dtype=dtype_c) for el in x]
        y = [el.to(dtype=dtype_c) for el in y]

        # --- Part 1: Embeddings ---
        if not self.patch_embedding:
            logging.debug("Loading patch_embedding")
            self._load_and_move_part_v1(
                "patch_embedding", nn.Conv3d,
                in_channels=self.in_dim, out_channels=self.dim,
                kernel_size=self.patch_size, stride=self.patch_size
            )

        device = self.patch_embedding.weight.device
        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # Move input data to the device for the operation
        x = [self.patch_embedding(u.unsqueeze(0).to(device, dtype=dtype_c)) for u in x]


        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        (f, h, w) = grid_sizes.tolist()[0]
        global freqs_i
        freqs_i = torch.cat([
            self.g_freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            self.g_freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            self.g_freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(seq_len, 1, -1)

        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        x = torch.cat(
            [torch.cat(
                [u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1) for u in x])  #+ 0.21Gb

        if not self.time_embedding:
            logging.debug("Loading time_embedding")
            self._load_and_move_part_v1("time_embedding", nn.Sequential,
                                     nnLinear2(self.freq_dim, self.dim),
                                     nn.SiLU(),
                                     nnLinear2(self.dim, self.dim))

        if not self.time_projection:
            logging.debug("Loading time_projection")
            self._load_and_move_part_v1("time_projection", nn.Sequential,
                                     nn.SiLU(),
                                     nnLinear2(self.dim, self.dim * 6))



        t_device = t.to(device)
        if t_device.dim() == 1:
            t_device = t_device.expand(t_device.size(0), seq_len)

        sin_embed = sinusoidal_embedding_1d(self.freq_dim,
                    t_device.flatten()).unflatten(0, (t_device.size(0), seq_len))
        target_dtype = self.time_embedding[0].weight.dtype
        sin_embed = sin_embed.to(device, dtype=target_dtype)


        e = self.time_embedding(sin_embed) # +0.21Gb


        del sin_embed

        #e0 = self.time_projection(e).unflatten(2, (6, self.dim))
        #self.register_buffer("e0", self.time_projection(e).unflatten(2, (6, self.dim)))

        e0 = self.time_projection(e).unflatten(2, (6, self.dim))
        if self.offload_large_tensors:
            e01 = e0[:, :, :3, :].to('cpu')
            e02 = e0[:, :, 3:, :].to('cpu')
            e = e.to("cpu")
        else:
            e01 = e0[:, :, :3, :]
            e02 = e0[:, :, 3:, :]

        del e0

        if not self.text_embedding:
            logging.debug("Loading text_embedding")
            self._load_and_move_part_v1("text_embedding", nn.Sequential,
                                     nnLinear2(self.text_dim, self.dim),
                                     nn.GELU(approximate='tanh'),
                                     nnLinear2(self.dim, self.dim))

        context = self.text_embedding(
            torch.stack(
                [torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                 for u in context]).to(device, dtype=dtype_c)
        )
        if context_null != None:
            context_null = self.text_embedding(
                torch.stack(
                    [torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                     for u in context_null]).to(device, dtype=dtype_c)
            )

        # --- Part 2: Main Blocks Loop ---
        x_null = x.clone()
        for i in range(self.num_layers):

            block_name = f"blocks.{i}"
            start_l = time.time_ns()
            if i < self.blocks_in_vram or i >= self.num_layers - self.blocks_in_ram:
                if i < self.blocks_in_vram:
                    logging.debug(f"Loading {block_name} VRAM")
                else:
                    logging.debug(f"Loading {block_name} RAM")
                self._load_and_move_part_v1(
                    block_name, WanAttentionBlock,
                    dim=self.dim, ffn_dim=self.ffn_dim, num_heads=self.num_heads,
                    window_size=self.window_size, qk_norm=self.qk_norm,
                    cross_attn_norm=self.cross_attn_norm, eps=self.eps
                )
                block = getattr(self, f"blocks{i}")
            else:
                logging.debug(f"Loading {block_name} DISC")
                self._load_and_move_part_v1(
                    block_name, WanAttentionBlock,
                    dim=self.dim, ffn_dim=self.ffn_dim, num_heads=self.num_heads,
                    window_size=self.window_size, qk_norm=self.qk_norm,
                    cross_attn_norm=self.cross_attn_norm, eps=self.eps
                )
                block = getattr(self, "blocks")
            end_l = time.time_ns()
            start_t = time.time_ns()

            x, x_null = block.do_attn(x, x_null, e01, seq_lens, grid_sizes, i)
            x, x_null = block.do_cross_attn(x, x_null, e02, context, context_null, None, i)

            end_t = time.time_ns()
            logging.debug(f"Inference time: {(end_t-start_t)/1000000} ms,  load time: {(end_l-start_l)/1000000} ms")

        # --- Part 3: Head and Unpatchify ---
        torch.cuda.empty_cache()
        if not self.head:
            logging.debug("Loading head...")
            self._load_and_move_part_v1(
                "head", Head,
                dim=self.dim,
                out_dim=self.out_dim,
                patch_size=self.patch_size,
                eps=self.eps
            )
        if self.offload_large_tensors:
            e = e.to("cuda")
        x = self.head(x, e)
        x_null = self.head(x_null, e)
        del e
        torch.cuda.empty_cache()

        logging.debug("Performing unpatchify...")
        x = self.unpatchify(x, grid_sizes)
        x_null = self.unpatchify(x_null, grid_sizes)

        del grid_sizes

        return [u
                for u in x], [u
                for u in x_null]

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out
