"""Python wrapper for decode_attn_bf16 (M=1 GQA decode attention with fused KV append)."""
import ctypes as ct
import math

import torch

SOURCE = "research/l2-prefetch/src/decode_attn.cu"
_module = None
_work = {}


def module():
    global _module
    if _module is None:
        from cuda_module import CudaModule
        _module = CudaModule((SOURCE,), ("decode_attn_bf16", "decode_attn_v2_bf16", "decode_attn_v3_bf16", "decode_attn_v4_bf16", "decode_attn_v5_bf16", "add_rmsnorm_bf16", "silu_mul_bf16", "decode_attn_v4d_bf16"), torch.cuda.current_device())
    return _module


def decode_attention(q, k_new, v_new, k_cache, v_cache, length, out=None, kernel="decode_attn_v4_bf16"):
    """q [heads*64], k_new/v_new [kv_heads*64], caches [kv_heads, max_len, 64] (BF16, contiguous).

    Writes k_new/v_new at position `length` and returns softmax(q K^T / 8) V over length+1 tokens.
    """
    kv_heads, max_len, hd = k_cache.shape
    heads = q.numel() // hd
    if hd != 64 or q.dtype != torch.bfloat16 or length + 1 > min(max_len, 4096) or heads % kv_heads:
        raise ValueError("Unsupported decode attention shape")
    for t in (q, k_new, v_new, k_cache, v_cache):
        if not t.is_contiguous() or t.data_ptr() % 16:
            raise ValueError("Tensors must be contiguous and 16-byte aligned")
    if out is None:
        out = torch.empty(heads * hd, device=q.device, dtype=torch.bfloat16)
    if kernel == "decode_attn_v5_bf16":
        if heads // kv_heads > 8:
            raise ValueError("v5 supports GQA groups up to 8")
        splits = (length + 1 + 63) // 64
        key = (q.device, heads, kv_heads, splits, "v5")
        if key not in _work:
            _work[key] = (torch.empty(heads * splits * (hd + 2), device=q.device, dtype=torch.float32),
                          torch.zeros(kv_heads, device=q.device, dtype=torch.int32))
        scratch, tickets = _work[key]
        module().launch(kernel, (kv_heads * splits, 1), 256, q, k_new, v_new, k_cache, v_cache, out, scratch, tickets,
                        ct.c_int(length), ct.c_int(max_len), ct.c_int(heads), ct.c_int(kv_heads),
                        ct.c_float(1.0 / math.sqrt(hd)), ct.c_int(splits))
        return out
    if kernel == "decode_attn_v4_bf16":
        splits = max(1, min(16, (length + 1 + 127) // 128))
        key = (q.device, heads, splits)
        if key not in _work:
            _work[key] = (torch.empty(heads * splits * (hd + 2), device=q.device, dtype=torch.float32),
                          torch.zeros(heads, device=q.device, dtype=torch.int32))
        scratch, tickets = _work[key]
        module().launch(kernel, (heads * splits, 1), 256, q, k_new, v_new, k_cache, v_cache, out, scratch, tickets,
                        ct.c_int(length), ct.c_int(max_len), ct.c_int(heads), ct.c_int(kv_heads),
                        ct.c_float(1.0 / math.sqrt(hd)), ct.c_int(splits))
        return out
    module().launch(kernel, (heads, 1), 512 if kernel.startswith("decode_attn_v3") else 256, q, k_new, v_new, k_cache, v_cache, out,
                    ct.c_int(length), ct.c_int(max_len), ct.c_int(heads), ct.c_int(kv_heads),
                    ct.c_float(1.0 / math.sqrt(hd)))
    return out


def add_rmsnorm(res, y, w, eps=1e-6):
    """Returns (x, xn) with x = res + y and xn = rms_norm(x) * w, both BF16 [M, d]."""
    m, d = res.shape
    if d % 2 or d > 2048 or res.dtype != torch.bfloat16:  # 8 values per thread in registers
        raise ValueError("Unsupported add_rmsnorm shape")
    x, xn = torch.empty_like(res), torch.empty_like(res)
    module().launch("add_rmsnorm_bf16", (m, 1), 256, res, y, w, x, xn, ct.c_int(d), ct.c_float(eps))
    return x, xn


def silu_mul(gu):
    """gu [M, 2*inter] BF16 -> silu(gate) * up [M, inter]."""
    m, two = gu.shape
    inter = two // 2
    out = torch.empty(m, inter, device=gu.device, dtype=gu.dtype)
    pairs = m * inter // 2
    module().launch("silu_mul_bf16", ((pairs + 255) // 256, 1), 256, gu, out, ct.c_int(inter), ct.c_int(pairs))
    return out


def decode_attention_dynamic(q, k_new, v_new, k_cache, v_cache, length, out=None):
    """Like decode_attention with the v4 kernel, but `length` is a CUDA int32 tensor holding the current cache length.

    The launch shape depends only on the cache capacity, so a captured CUDA graph stays valid while the caller
    increments `length` between replays (length + 1 must stay <= capacity).
    """
    kv_heads, max_len, hd = k_cache.shape
    heads = q.numel() // hd
    if hd != 64 or q.dtype != torch.bfloat16 or heads % kv_heads or length.dtype != torch.int32 or length.numel() != 1:
        raise ValueError("Unsupported dynamic decode attention arguments")
    splits = max(1, min(32, (max_len + 127) // 128))
    if (max_len + splits - 1) // splits > 4096:
        raise ValueError("Cache capacity too large for the per-split score buffer")
    key = (q.device, heads, splits, "v4d")
    if key not in _work:
        _work[key] = (torch.empty(heads * splits * (hd + 2), device=q.device, dtype=torch.float32),
                      torch.zeros(heads, device=q.device, dtype=torch.int32))
    scratch, tickets = _work[key]
    if out is None:
        out = torch.empty(heads * hd, device=q.device, dtype=torch.bfloat16)
    module().launch("decode_attn_v4d_bf16", (heads * splits, 1), 256, q, k_new, v_new, k_cache, v_cache, out,
                    scratch, tickets, length, ct.c_int(max_len), ct.c_int(heads), ct.c_int(kv_heads),
                    ct.c_float(1.0 / math.sqrt(hd)), ct.c_int(splits))
    return out
