# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.

# changed to run inference on 8GB VRAM
# (generated frames limited to 21, increase if you have more VRAM)
import gc
import math
import torch
import time
import torch.nn as nn
import safetensors.torch

from typing import List

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

from .attention import flash_attention

from torch.nn import functional as F, init
from torch import Tensor

#from line_profiler import profile
#from memory_profiler import profile


def print_gpu_memory_report():
    """
    Prints a comprehensive report of GPU memory usage, including a summary from
    PyTorch's memory management and a detailed list of all CUDA tensors.
    """

    # 1. PyTorch's built-in memory summary
    print("--- PyTorch CUDA Memory Summary ---")
    # torch.cuda.memory_summary() provides a detailed, human-readable report
    # on memory allocation, fragmentation, and caching.
    print(torch.cuda.memory_summary())

    # You can also get the raw numbers
    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)
    print(f"Current Memory (Allocated by Tensors): {allocated:.2f} GB")
    print(f"Cached Memory (Reserved by PyTorch):   {reserved:.2f} GB")
    print("-" * 40)
    print("\n")

    # 2. Detailed Tensor Breakdown (your original function, slightly modified)
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
        dtype=torch.bfloat16,
    ) -> None:
        dtype = torch.bfloat16
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

    #@profile
    def forward(self, input: Tensor) -> Tensor:
        #w = self.weight.clone()#.detach()
        #b = self.bias.clone()#.detach()
        #return F.linear(input, w, b)

        return F.linear(input, self.weight, self.bias)

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
                        return torch.nn.Parameter(p.to(**kwargs), requires_grad=p.requires_grad)
                    else:
                        return p.to(**kwargs)
            if '_buffers' in self.__dict__:
                _buffers = self.__dict__['_buffers']
                if name in _buffers:
                    return _buffers[name].to(**kwargs)
            return super(original_class, self).__getattr__(name)

        module.__class__ = type('DynamicSwap_' + original_class.__name__, (original_class,), {
            '__getattr__': hacked_get_attr,
        })

        return

    @staticmethod
    def install_model(model: torch.nn.Module, **kwargs):
        count_modules = 0
        for m in model.modules():
            count_modules += 1
            DynamicSwapInstaller._install_module(m, **kwargs)
        print("installed modules " + str(count_modules) + "\n")
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


#@torch.amp.autocast('cuda', enabled=False)
def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta,
                        torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


#@torch.amp.autocast('cuda', enabled=False)
#@profile
def rope_apply(x, grid_sizes, freqs):
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
        output.append(x_i.to(torch.bfloat16))
    return torch.stack(output)


class WanRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = torch.tensor(eps, dtype=torch.bfloat16, device="cuda")
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.bfloat16))

    #@profile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(
            x.pow(2).mean(-1, keepdim=True, dtype=torch.bfloat16) + self.eps,
        ) * self.weight


class WanLayerNorm(nn.LayerNorm):

    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return super().forward(x).type_as(x)


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
        self.q = nnLinear(dim, dim, dtype=torch.bfloat16)
        self.k = nnLinear(dim, dim, dtype=torch.bfloat16)
        self.v = nnLinear(dim, dim, dtype=torch.bfloat16)
        self.o = nnLinear(dim, dim, dtype=torch.bfloat16)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    #@profile
    def forward(self, x, seq_lens, grid_sizes, freqs):
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
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        x = flash_attention(
            q=rope_apply(q, grid_sizes, freqs),
            k=rope_apply(k, grid_sizes, freqs),
            v=v,
            k_lens=seq_lens,
            window_size=self.window_size)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanCrossAttention(WanSelfAttention):

    #@profile
    def forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)

        # compute attention
        x = flash_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        x = self.o(x)
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
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm,
                                          eps)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WanCrossAttention(dim, num_heads, (-1, -1), qk_norm,
                                            eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nnLinear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nnLinear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    #@profile
    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
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
        #assert e.dtype == torch.float32
        #with torch.amp.autocast('cuda', dtype=torch.float32):
        e = (self.modulation.unsqueeze(0) + e).chunk(6, dim=2)
        #assert e[0].dtype == torch.float32

        # self-attention
        y = self.self_attn(
            self.norm1(x) * (1 + e[1].squeeze(2)) + e[0].squeeze(2),
            seq_lens, grid_sizes, freqs)
        #with torch.amp.autocast('cuda', dtype=torch.float32):
        x = x + y * e[2].squeeze(2)

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e):
            x = x + self.cross_attn(self.norm3(x), context, context_lens)
            y = self.ffn(
                self.norm2(x) * (1 + e[4].squeeze(2)) + e[3].squeeze(2))
            #with torch.amp.autocast('cuda', dtype=torch.float32):
            x = x + y * e[5].squeeze(2)
            return x

        x = cross_attn_ffn(x, context, context_lens, e)
        return x


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
        self.head = nnLinear(dim, out_dim, dtype=torch.bfloat16)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim, dtype=torch.bfloat16) / dim**0.5)
        self.mod_0 = None
        self.mod_1 = None

    def set_modulation(self):
        mod_0, mod_1 = self.modulation.chunk(2, dim=1)
        self.mod_0 = mod_0.squeeze(1).to("cuda", dtype=torch.bfloat16)
        self.mod_1 = mod_1.squeeze(1).to("cuda", dtype=torch.bfloat16)

    def forward(self, x, e):
        if self.mod_0 is None:
            self.set_modulation()

        return self.head(self.norm(x) * (1 + self.mod_1 + e) + self.mod_0 + e)


g_freqs = None


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
        for i in range(self.blocks_count):
            if getattr(self, f"blocks{i}"):
                delattr(self, f"blocks{i}")

        self.patch_embedding = None
        self.text_embedding = None
        self.time_embedding = None
        self.time_projection = None
        self.head = None
        gc.collect()
        torch.cuda.empty_cache()

    def _load_and_move_part_v1(self, part_name: str, part_class, *args, **kwargs):
        file_path = self.model_path + part_name + '.safetensors'
        attr_name = part_name.replace(".", "")

        if getattr(self, attr_name):
            return

        part_state_dict = safetensors.torch.load_file(file_path, device="cpu")

        if self.blocks_in_vram > 0 and part_name.split(".")[0] == "blocks" \
                and int(part_name.split(".")[1]) < self.blocks_in_vram:
            if getattr(self, attr_name) == None:
                with torch.device("meta"):
                    layer = part_class(*args, **kwargs)
                layer = layer.to_empty(device="cpu")

                layer.load_state_dict(part_state_dict, assign=True)
                layer.to('cuda', non_blocking=True)
                setattr(self, attr_name, layer)
            return

        if part_name.split(".")[0] == "blocks" \
                and int(part_name.split(".")[1]) >= self.blocks_count - self.blocks_in_ram:
            with torch.device("meta"):
                layer = part_class(*args, **kwargs)
            layer = layer.to_empty(device="cpu")

            layer.load_state_dict(part_state_dict, assign=True)
            setattr(self, attr_name, layer)
            DynamicSwapInstaller.install_model(getattr(self, attr_name), device="cuda")
        else:
            if part_name.split(".")[0] == "blocks":
                attr_name = "blocks"
            if getattr(self, attr_name):
                getattr(self, attr_name).load_state_dict(part_state_dict, assign=True)
            else:
                with torch.device("meta"):
                    layer = part_class(*args, **kwargs)
                layer = layer.to_empty(device="cpu")
                layer.load_state_dict(part_state_dict, assign=True)
                if part_name == "head":
                    layer.to("cuda")

                setattr(self, attr_name, layer)
                if part_name != "head":
                    DynamicSwapInstaller.install_model(getattr(self, attr_name), device="cuda")

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
                 blocks_in_vram=0):
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
        self.in_dim = in_dim
        if model_type == 'i2v_h' or model_type == 'i2v_l':
            self.in_dim = 36

        if model_type == 't2v_h':
            # todo: replace with your path to converted models see: optimize_files.py
            self.model_path = "./Wan2.2-T2V-A14B/high_noise_model/"
            self.blocks_count = 40

        if model_type == 't2v_l':
            # todo: replace with your path to converted models see: optimize_files.py
            self.model_path = "./Wan2.2-T2V-A14B/low_noise_model/"
            self.blocks_count = 40

        if model_type == 'i2v_h':
            # todo: replace with your path to converted models see: optimize_files.py
            self.model_path = "./Wan2.2-I2V-A14B/high_noise_model/"
            self.blocks_count = 40

        if model_type == 'i2v_l':
            # todo: replace with your path to converted models see: optimize_files.py
            self.model_path = "./Wan2.2-I2V-A14B/low_noise_model/"
            self.blocks_count = 40

        if model_type == 'ti2v':
            self.model_path = "./Wan2.2-TI2V-5B/model/"
            self.blocks_count = 30

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
        self.blocks0 = None
        self.blocks1 = None
        self.blocks2 = None
        self.blocks3 = None
        self.blocks4 = None
        self.blocks5 = None
        self.blocks6 = None
        self.blocks7 = None
        self.blocks8 = None
        self.blocks9 = None
        self.blocks10 = None
        self.blocks11 = None
        self.blocks12 = None
        self.blocks13 = None
        self.blocks14 = None
        self.blocks15 = None
        self.blocks16 = None
        self.blocks17 = None
        self.blocks18 = None
        self.blocks19 = None
        self.blocks20 = None
        self.blocks21 = None
        self.blocks22 = None
        self.blocks23 = None
        self.blocks24 = None
        self.blocks25 = None
        self.blocks26 = None
        self.blocks27 = None
        self.blocks28 = None
        self.blocks29 = None
        self.blocks30 = None
        self.blocks31 = None
        self.blocks32 = None
        self.blocks33 = None
        self.blocks34 = None
        self.blocks35 = None
        self.blocks36 = None
        self.blocks37 = None
        self.blocks38 = None
        self.blocks39 = None

        self.head = None

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ], dim=1)

        # precompute g_freqs
        c = 64
        global g_freqs
        g_freqs = self.freqs.to("cuda").split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

        self.blocks_in_ram = blocks_in_ram
        print(f"\nRAM available for {self.blocks_in_ram}/{self.blocks_count} blocks\n")

        self.blocks_in_vram = blocks_in_vram
        print(f"\nVRAM available for {self.blocks_in_vram}/{self.blocks_count} blocks\n")

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
        # --- Part 1: Embeddings ---
        if not self.patch_embedding:
            print("Loading patch_embedding")
            self._load_and_move_part_v1(
                "patch_embedding", nn.Conv3d,
                in_channels=self.in_dim, out_channels=self.dim,
                kernel_size=self.patch_size, stride=self.patch_size
            )

        device = self.patch_embedding.weight.device
        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # Move input data to the device for the operation
        x = [self.patch_embedding(u.unsqueeze(0).to(device, dtype=torch.bfloat16)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        x = torch.cat(
            [torch.cat(
                [u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1) for u in x])

        if not self.time_embedding:
            print("Loading time_embedding")
            self._load_and_move_part_v1("time_embedding", nn.Sequential,
                                     nnLinear(self.freq_dim, self.dim),
                                     nn.SiLU(),
                                     nnLinear(self.dim, self.dim))
        if not self.time_projection:
            print("Loading time_projection")
            self._load_and_move_part_v1("time_projection", nn.Sequential,
                                     nn.SiLU(),
                                     nnLinear(self.dim, self.dim * 6))

        t_device = t.to(device)
        if t_device.dim() == 1:
            t_device = t_device.expand(t_device.size(0), seq_len)
        sin_embed = sinusoidal_embedding_1d(self.freq_dim,
                    t_device.flatten()).unflatten(0, (t_device.size(0), seq_len))
        target_dtype = self.time_embedding[0].weight.dtype
        sin_embed = sin_embed.to(device, dtype=target_dtype)

        e = self.time_embedding(sin_embed)

        del sin_embed

        e0 = self.time_projection(e).unflatten(2, (6, self.dim))

        if not self.text_embedding:
            print("Loading text_embedding")
            self._load_and_move_part_v1("text_embedding", nn.Sequential,
                                     nnLinear(self.text_dim, self.dim),
                                     nn.GELU(approximate='tanh'),
                                     nnLinear(self.dim, self.dim))

        context = self.text_embedding(
            torch.stack(
                [torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                 for u in context]).to(device)
        )
        if context_null != None:
            context_null = self.text_embedding(
                torch.stack(
                    [torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                     for u in context_null]).to(device)
            )

        # --- Part 2: Main Blocks Loop ---
        kwargs = dict(e=e0, seq_lens=seq_lens, grid_sizes=grid_sizes,
                      #freqs=self.freqs.to(device),
                      freqs=None,
                      context=context, context_lens=None)
        if context_null != None:
            kwargs_null = dict(e=e0, seq_lens=seq_lens, grid_sizes=grid_sizes,
                               #freqs=self.freqs.to(device),
                               freqs=None,
                               context=context_null, context_lens=None)
        else:
            kwargs_null = None

        del seq_lens, context, context_null, e0

        x_null = x.clone()
        for i in range(self.num_layers):
            block_name = f"blocks.{i}"
            start_l = time.time_ns()
            if i < self.blocks_in_vram or i >= self.blocks_count - self.blocks_in_ram:
                if i < self.blocks_in_vram:
                    print(f"Loading {block_name} VRAM")
                else:
                    print(f"Loading {block_name} RAM")
                self._load_and_move_part_v1(
                    block_name, WanAttentionBlock,
                    dim=self.dim, ffn_dim=self.ffn_dim, num_heads=self.num_heads,
                    window_size=self.window_size, qk_norm=self.qk_norm,
                    cross_attn_norm=self.cross_attn_norm, eps=self.eps
                )
                block = getattr(self, f"blocks{i}")
            else:
                print(f"Loading {block_name} DISC")
                self._load_and_move_part_v1(
                    block_name, WanAttentionBlock,
                    dim=self.dim, ffn_dim=self.ffn_dim, num_heads=self.num_heads,
                    window_size=self.window_size, qk_norm=self.qk_norm,
                    cross_attn_norm=self.cross_attn_norm, eps=self.eps
                )
                block = getattr(self, "blocks")
            end_l = time.time_ns()
            start_t = time.time_ns()
            x = block(x, **kwargs)
            if kwargs_null:
                x_null = block(x_null, **kwargs_null)
            else:
                x_null = None
            end_t = time.time_ns()
            print(f"Inference time: {(end_t-start_t)/1000000} ms,  load time: {(end_l-start_l)/1000000} ms")

        # --- Part 3: Head and Unpatchify ---

        if not self.head:
            print("Loading head to CUDA...")
            self._load_and_move_part_v1(
                "head", Head,
                dim=self.dim,
                out_dim=self.out_dim,
                patch_size=self.patch_size,
                eps=self.eps
            )
        x = self.head(x, e)
        if kwargs_null:
            x_null = self.head(x_null, e)

        del e

        print("Performing unpatchify...")
        x = self.unpatchify(x, grid_sizes)
        if kwargs_null:
            x_null = self.unpatchify(x_null, grid_sizes)
        else:
            x_null = []

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
