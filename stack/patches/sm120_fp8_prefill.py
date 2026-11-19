"""Opt-in vLLM 0.34 patch: route causal prefill with an FP8 KV cache through FlashInfer's SM120 FP8 FMHA kernel
(FP8 Q/K/V on the tensor cores) instead of the FA2 bf16-Q kernel. Adapted from sm120-vllm-kernels patch 01 (vLLM 0.32);
the 0.32 patch does not apply to 0.34 (renamed locals, no wrapper._max_q_len), so the hook is re-expressed here.
Off unless VLLM_SM120_FP8_PREFILL=1. Usage: python3 sm120_fp8_prefill.py [apply|revert]
"""
import pathlib, shutil, sys

F = pathlib.Path("/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/flashinfer.py")
BAK = F.with_suffix(".py.orig-sm120fp8")

HELPER = '''
# --- secai: SM120 FP8 prefill (opt-in, VLLM_SM120_FP8_PREFILL=1) ---
import os as _os

_SM120_FP8_PREFILL = _os.environ.get("VLLM_SM120_FP8_PREFILL", "0") == "1"
# Spec-decode verification steps (query len 1+k) also take the prefill path and must keep the bf16 kernel.
_SM120_FP8_MIN_Q = int(_os.environ.get("VLLM_SM120_FP8_PREFILL_MIN_Q", "256"))
_FMHA_FN_CACHE: dict = {}


def _sm120_fmha_fn(hq, hkv, d, page, out_dtype, device):
    key = (hq, hkv, d, page, out_dtype)
    fn = _FMHA_FN_CACHE.get(key)
    if fn is None:
        from flashinfer.attention.cute_dsl.sm120_fmha import _use_balanced_scheduler
        from flashinfer.cute_dsl.attention.fmha.sm120 import (
            SM120FusedMultiHeadAttentionFP8ForwardTMA as K,
            compile_sm120_fmha_fp8_paged_kernel,
        )

        fn = compile_sm120_fmha_fp8_paged_kernel(
            in_dtype=torch.float8_e4m3fn, out_dtype=out_dtype, num_qo_heads=hq, num_kv_heads=hkv,
            head_dim=d, is_causal=True, kv_tile=K.SEQ_KV_TILES[0], q_tile=K.SEQ_Q_TILES[0],
            num_tokens_per_page=page, device=device, with_lse=False,
            balanced_scheduler=_use_balanced_scheduler(True),
        )
        _FMHA_FN_CACHE[key] = fn
    return fn


def _sm120_fp8_meta(attn_metadata, wrapper, page):
    """Block table / lengths depend only on the step's plan: build once per step, share across layers."""
    meta = getattr(attn_metadata, "_sm120_fp8_meta", None)
    if meta is None:
        from cutlass.cute.typing import Int32

        idx = wrapper._paged_kv_indices_buf
        indptr = wrapper._paged_kv_indptr_buf
        last = wrapper._paged_kv_last_page_len_buf
        qo = wrapper._qo_indptr_buf.to(torch.int32)
        n_pages = indptr[1:] - indptr[:-1]
        seqlens = ((n_pages - 1) * page + last).to(torch.int32)
        max_pages = int(n_pages.max())
        bt = indptr[:-1, None] + torch.arange(max_pages, device=idx.device, dtype=indptr.dtype)[None, :]
        bt = bt.clamp_(max=max(idx.shape[0] - 1, 0)).to(torch.int32)
        max_q = int((qo[1:] - qo[:-1]).max())
        meta = (idx, bt, seqlens, qo, Int32(max_q), max_q)
        attn_metadata._sm120_fp8_meta = meta
    return meta


def _sm120_fp8_paged_prefill(impl, layer, attn_metadata, wrapper, query, kv_cache_tuple, out):
    from cutlass.cute.typing import Float32

    k, v = kv_cache_tuple
    if getattr(wrapper, "_kv_layout", "NHD") == "NHD":
        k, v = k.transpose(1, 2), v.transpose(1, 2)
    page = k.shape[2]
    idx, bt, seqlens, cu_q, max_q, _ = _sm120_fp8_meta(attn_metadata, wrapper, page)
    # The kernel needs compact [Hkv, page, D] planes; vLLM packs K|V per row, so gather this batch's pages.
    kc = k.index_select(0, idx)
    vc = v.index_select(0, idx)
    q_scale = layer._q_scale_float
    q = query if q_scale == 1.0 else query / q_scale
    q8 = q.clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    if out.dim() == 2:
        out = out.view(out.shape[0], -1, query.shape[-1])
    fn = _sm120_fmha_fn(q8.shape[1], kc.shape[1], q8.shape[2], page, out.dtype, q8.device)
    sm_scale = impl.scale * q_scale * layer._k_scale_float
    # FlashInfer 0.7.4 ABI: q, k, v, o, lse, scale_log2, output_scale, skip_softmax_threshold, seqlens_kv, cu_seqlens_q,
    # block_tables, cu_seqlens_k, max_seqlen_q, use_pdl, kv_tile_ranges
    fn(q8, kc, vc, out, None, Float32(sm_scale * 1.4426950408889634), Float32(layer._v_scale_float),
       None, seqlens, cu_q, bt, None, max_q, False, None)
# --- end secai ---
'''

ANCHOR1 = "logger = init_logger(__name__)\n"
OLD = """                    else:
                        prefill_wrapper.run(
                            prefill_query,
                            kv_cache_for_fi,
                            q_scale=self.get_q_scale(layer, prefill_query.dtype),"""
NEW = """                    elif (
                        _SM120_FP8_PREFILL
                        and not self.is_kvcache_nvfp4
                        and kv_cache_tuple[0].dtype == torch.float8_e4m3fn
                        and attn_metadata.causal
                        and self.window_left == -1
                        and not self.logits_soft_cap
                        and not needs_fp8_out_prefill
                        and _sm120_fp8_meta(attn_metadata, prefill_wrapper, kv_cache_tuple[0].shape[1 if prefill_wrapper._kv_layout == "NHD" else 2])[5] >= _SM120_FP8_MIN_Q
                    ):
                        _sm120_fp8_paged_prefill(
                            self, layer, attn_metadata, prefill_wrapper, prefill_query, kv_cache_tuple, out_prefill
                        )
                    else:
                        prefill_wrapper.run(
                            prefill_query,
                            kv_cache_for_fi,
                            q_scale=self.get_q_scale(layer, prefill_query.dtype),"""


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "apply"
    if mode == "revert":
        shutil.copy(BAK, F); print("reverted"); return
    src = F.read_text()
    if "_sm120_fp8_paged_prefill" in src:  # re-apply on top of the pristine copy
        shutil.copy(BAK, F); src = F.read_text()
    assert src.count(ANCHOR1) == 1 and src.count(OLD) == 1, "anchors not found: vLLM source changed"
    if not BAK.exists():
        shutil.copy(F, BAK)
    F.write_text(src.replace(ANCHOR1, ANCHOR1 + HELPER, 1).replace(OLD, NEW, 1))
    print("applied")


if __name__ == "__main__":
    main()
