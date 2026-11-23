import torch
import torch.nn as nn

from einops import rearrange

try:
    from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func
    from flash_attn.bert_padding import unpad_input, pad_input
    HAS_FLASH_ATTN = True
    FLASH_ATTN_IMPORT_ERROR = None
except Exception as exc:
    flash_attn_varlen_qkvpacked_func = None
    unpad_input = None
    pad_input = None
    HAS_FLASH_ATTN = False
    FLASH_ATTN_IMPORT_ERROR = exc


class FlashAttention(nn.Module):
    """Implement scaled dot product attention with flash-attn when available."""

    def __init__(self, softmax_scale=None, attention_dropout=0.0, device=None, dtype=None):
        super().__init__()
        self.softmax_scale = softmax_scale
        self.dropout_p = attention_dropout

    @staticmethod
    def available():
        return HAS_FLASH_ATTN

    def forward(self, qkv, key_padding_mask=None, causal=False, cu_seqlens=None,
                max_s=None, need_weights=False):
        """Implements multi-head softmax attention via flash-attn kernels."""
        if not HAS_FLASH_ATTN:
            raise RuntimeError(
                "FlashAttention was requested, but flash_attn is not installed "
                "or cannot be imported in this environment. Set "
                "use_flash_attn/use_fused_rmsnorm/use_fused_mlp to False. "
                "Original import error: {}".format(FLASH_ATTN_IMPORT_ERROR))

        assert not need_weights
        assert qkv.dtype in [torch.float16, torch.bfloat16]
        assert qkv.is_cuda

        if cu_seqlens is None:
            batch_size = qkv.shape[0]
            seqlen = qkv.shape[1]
            if key_padding_mask is None:
                qkv = rearrange(qkv, 'b s ... -> (b s) ...')
                max_s = seqlen
                cu_seqlens = torch.arange(
                    0, (batch_size + 1) * seqlen, step=seqlen,
                    dtype=torch.int32, device=qkv.device)
                output = flash_attn_varlen_qkvpacked_func(
                    qkv, cu_seqlens, max_s,
                    self.dropout_p if self.training else 0.0,
                    softmax_scale=self.softmax_scale, causal=causal)
                output = rearrange(output, '(b s) ... -> b s ...', b=batch_size)
            else:
                nheads = qkv.shape[-2]
                x = rearrange(qkv, 'b s three h d -> b s (three h d)')
                x_unpad, indices, cu_seqlens, max_s = unpad_input(x, key_padding_mask)
                x_unpad = rearrange(
                    x_unpad, 'nnz (three h d) -> nnz three h d', three=3, h=nheads)
                output_unpad = flash_attn_varlen_qkvpacked_func(
                    x_unpad, cu_seqlens, max_s,
                    self.dropout_p if self.training else 0.0,
                    softmax_scale=self.softmax_scale, causal=causal)
                output = rearrange(
                    pad_input(rearrange(output_unpad, 'nnz h d -> nnz (h d)'),
                              indices, batch_size, seqlen),
                    'b s (h d) -> b s h d', h=nheads)
        else:
            assert max_s is not None
            output = flash_attn_varlen_qkvpacked_func(
                qkv, cu_seqlens, max_s,
                self.dropout_p if self.training else 0.0,
                softmax_scale=self.softmax_scale, causal=causal)

        return output, None
