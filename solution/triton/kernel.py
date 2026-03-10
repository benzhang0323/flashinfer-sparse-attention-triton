from __future__ import annotations

import torch
import triton
import triton.language as tl

NUM_QO_HEADS = 16
HEAD_DIM_CKV = 512
HEAD_DIM_KPE = 64
PAGE_SIZE = 64
TOPK = 2048

# ---------------------------
# Locked config (B200 tuned)
# ---------------------------
BLOCK_K_FIXED = 64
BD_FIXED = 64
BDP_FIXED = 32
NUM_WARPS_FIXED = 4
NUM_STAGES_FIXED = 1          # launch-time stages
LOOP_STAGES_FIXED = 2         # tl.range() pipelining stages (B200)


@triton.jit
def _kernel(
    qn_ptr,
    qp_ptr,
    ckv_ptr,
    kpe_ptr,
    si_ptr,
    sm_scale_ptr,
    out_ptr,
    lse_ptr,
    # runtime-ish scalars
    num_tokens: tl.constexpr,
    kv_capacity: tl.constexpr,
    num_heads: tl.constexpr,
    # strides
    s_qn_t: tl.constexpr,
    s_qn_h: tl.constexpr,
    s_qn_d: tl.constexpr,
    s_qp_t: tl.constexpr,
    s_qp_h: tl.constexpr,
    s_qp_d: tl.constexpr,
    s_ckv_p: tl.constexpr,
    s_ckv_s: tl.constexpr,
    s_ckv_d: tl.constexpr,
    s_kpe_p: tl.constexpr,
    s_kpe_s: tl.constexpr,
    s_kpe_d: tl.constexpr,
    s_si_t: tl.constexpr,
    s_si_k: tl.constexpr,
    s_out_t: tl.constexpr,
    s_out_h: tl.constexpr,
    s_out_d: tl.constexpr,
    s_lse_t: tl.constexpr,
    s_lse_h: tl.constexpr,
    HAS_LSE: tl.constexpr,
    # output dtype selection
    OUT_DTYPE: tl.constexpr,
    # meta
    BLOCK_K: tl.constexpr,
    BD: tl.constexpr,
    BDP: tl.constexpr,
    LOOP_STAGES: tl.constexpr,
):
    TOPK_L = 2048
    D_CKV_L = 512
    D_KPE_L = 64
    INV_LN2_L = 1.4426950408889634  # 1/ln(2)

    pid = tl.program_id(0)
    t = pid // num_heads
    h = pid - t * num_heads

    inb = (t < num_tokens) & (h < num_heads)

    sm_scale = tl.load(sm_scale_ptr).to(tl.float32)

    m = tl.full((), -float("inf"), tl.float32)
    l = tl.full((), 0.0, tl.float32)

    acc0 = tl.zeros([BD], tl.float32)
    acc1 = tl.zeros([BD], tl.float32)
    acc2 = tl.zeros([BD], tl.float32)
    acc3 = tl.zeros([BD], tl.float32)
    acc4 = tl.zeros([BD], tl.float32)
    acc5 = tl.zeros([BD], tl.float32)
    acc6 = tl.zeros([BD], tl.float32)
    acc7 = tl.zeros([BD], tl.float32)

    # Strides as int64 once
    sckvp = tl.full((), s_ckv_p, tl.int64)
    sckvs = tl.full((), s_ckv_s, tl.int64)
    sckvd = tl.full((), s_ckv_d, tl.int64)

    skpep = tl.full((), s_kpe_p, tl.int64)
    skpes = tl.full((), s_kpe_s, tl.int64)
    skped = tl.full((), s_kpe_d, tl.int64)

    cap = tl.full((), kv_capacity, tl.int32)

    # Preload q_pe in 2x32
    offs_bdp = tl.arange(0, BDP)
    qp0 = tl.load(
        qp_ptr + t * s_qp_t + h * s_qp_h + (0 * BDP + offs_bdp) * s_qp_d,
        mask=inb & (offs_bdp < D_KPE_L),
        other=0,
    ).to(tl.float32)
    qp1 = tl.load(
        qp_ptr + t * s_qp_t + h * s_qp_h + (1 * BDP + offs_bdp) * s_qp_d,
        mask=inb & ((BDP + offs_bdp) < D_KPE_L),
        other=0,
    ).to(tl.float32)

    offs_bd = tl.arange(0, BD)

    for k0 in tl.range(
        0,
        TOPK_L,
        BLOCK_K,
        num_stages=LOOP_STAGES,
        loop_unroll_factor=1,
        disable_licm=True,
    ):
        kk = k0 + tl.arange(0, BLOCK_K)
        k_mask = kk < TOPK_L

        kv_i32 = tl.load(
            si_ptr + t * s_si_t + kk * s_si_k,
            mask=inb & k_mask,
            other=-1,
        ).to(tl.int32)

        valid = inb & k_mask & (kv_i32 != -1) & (kv_i32 >= 0) & (kv_i32 < cap)
        kv_safe = tl.where(valid, kv_i32, 0).to(tl.int64)

        # PAGE_SIZE==64 => shifts/masks
        page = kv_safe >> 6
        off = kv_safe & 63

        row_ckv = page * sckvp + off * sckvs
        row_kpe = page * skpep + off * skpes

        # ----- logits = q_nope·Kc + q_pe·Kp -----
        score_c = tl.zeros([BLOCK_K], tl.float32)

        d = 0 * BD + offs_bd
        qn = tl.load(
            qn_ptr + t * s_qn_t + h * s_qn_h + d * s_qn_d,
            mask=inb,
            other=0,
        ).to(tl.float32)
        kc = tl.load(
            ckv_ptr + row_ckv[:, None] + d[None, :].to(tl.int64) * sckvd,
            mask=valid[:, None],
            other=0,
            cache_modifier=".cg",
            eviction_policy="evict_last",
        ).to(tl.float32)
        score_c += tl.sum(kc * qn[None, :], axis=1)

        d = 1 * BD + offs_bd
        qn = tl.load(
            qn_ptr + t * s_qn_t + h * s_qn_h + d * s_qn_d,
            mask=inb,
            other=0,
        ).to(tl.float32)
        kc = tl.load(
            ckv_ptr + row_ckv[:, None] + d[None, :].to(tl.int64) * sckvd,
            mask=valid[:, None],
            other=0,
            cache_modifier=".cg",
            eviction_policy="evict_last",
        ).to(tl.float32)
        score_c += tl.sum(kc * qn[None, :], axis=1)

        d = 2 * BD + offs_bd
        qn = tl.load(
            qn_ptr + t * s_qn_t + h * s_qn_h + d * s_qn_d,
            mask=inb,
            other=0,
        ).to(tl.float32)
        kc = tl.load(
            ckv_ptr + row_ckv[:, None] + d[None, :].to(tl.int64) * sckvd,
            mask=valid[:, None],
            other=0,
            cache_modifier=".cg",
            eviction_policy="evict_last",
        ).to(tl.float32)
        score_c += tl.sum(kc * qn[None, :], axis=1)

        d = 3 * BD + offs_bd
        qn = tl.load(
            qn_ptr + t * s_qn_t + h * s_qn_h + d * s_qn_d,
            mask=inb,
            other=0,
        ).to(tl.float32)
        kc = tl.load(
            ckv_ptr + row_ckv[:, None] + d[None, :].to(tl.int64) * sckvd,
            mask=valid[:, None],
            other=0,
            cache_modifier=".cg",
            eviction_policy="evict_last",
        ).to(tl.float32)
        score_c += tl.sum(kc * qn[None, :], axis=1)

        d = 4 * BD + offs_bd
        qn = tl.load(
            qn_ptr + t * s_qn_t + h * s_qn_h + d * s_qn_d,
            mask=inb,
            other=0,
        ).to(tl.float32)
        kc = tl.load(
            ckv_ptr + row_ckv[:, None] + d[None, :].to(tl.int64) * sckvd,
            mask=valid[:, None],
            other=0,
            cache_modifier=".cg",
            eviction_policy="evict_last",
        ).to(tl.float32)
        score_c += tl.sum(kc * qn[None, :], axis=1)

        d = 5 * BD + offs_bd
        qn = tl.load(
            qn_ptr + t * s_qn_t + h * s_qn_h + d * s_qn_d,
            mask=inb,
            other=0,
        ).to(tl.float32)
        kc = tl.load(
            ckv_ptr + row_ckv[:, None] + d[None, :].to(tl.int64) * sckvd,
            mask=valid[:, None],
            other=0,
            cache_modifier=".cg",
            eviction_policy="evict_last",
        ).to(tl.float32)
        score_c += tl.sum(kc * qn[None, :], axis=1)

        d = 6 * BD + offs_bd
        qn = tl.load(
            qn_ptr + t * s_qn_t + h * s_qn_h + d * s_qn_d,
            mask=inb,
            other=0,
        ).to(tl.float32)
        kc = tl.load(
            ckv_ptr + row_ckv[:, None] + d[None, :].to(tl.int64) * sckvd,
            mask=valid[:, None],
            other=0,
            cache_modifier=".cg",
            eviction_policy="evict_last",
        ).to(tl.float32)
        score_c += tl.sum(kc * qn[None, :], axis=1)

        d = 7 * BD + offs_bd
        qn = tl.load(
            qn_ptr + t * s_qn_t + h * s_qn_h + d * s_qn_d,
            mask=inb,
            other=0,
        ).to(tl.float32)
        kc = tl.load(
            ckv_ptr + row_ckv[:, None] + d[None, :].to(tl.int64) * sckvd,
            mask=valid[:, None],
            other=0,
            cache_modifier=".cg",
            eviction_policy="evict_last",
        ).to(tl.float32)
        score_c += tl.sum(kc * qn[None, :], axis=1)

        score_p = tl.zeros([BLOCK_K], tl.float32)

        dp = 0 * BDP + offs_bdp
        kp = tl.load(
            kpe_ptr + row_kpe[:, None] + dp[None, :].to(tl.int64) * skped,
            mask=valid[:, None],
            other=0,
            cache_modifier=".cg",
            eviction_policy="evict_last",
        ).to(tl.float32)
        score_p += tl.sum(kp * qp0[None, :], axis=1)

        dp = 1 * BDP + offs_bdp
        kp = tl.load(
            kpe_ptr + row_kpe[:, None] + dp[None, :].to(tl.int64) * skped,
            mask=valid[:, None],
            other=0,
            cache_modifier=".cg",
            eviction_policy="evict_last",
        ).to(tl.float32)
        score_p += tl.sum(kp * qp1[None, :], axis=1)

        logits = (score_c + score_p) * sm_scale
        logits = tl.where(valid, logits, -float("inf"))

        block_m = tl.max(logits, axis=0)
        has_any = tl.sum(valid, axis=0) > 0
        block_m = tl.where(has_any, block_m, -float("inf"))
        m_new = tl.maximum(m, block_m)

        alpha = tl.where(m == -float("inf"), 0.0, tl.exp(m - m_new))
        alpha = tl.where(has_any, alpha, 1.0)

        l = l * alpha
        acc0 *= alpha
        acc1 *= alpha
        acc2 *= alpha
        acc3 *= alpha
        acc4 *= alpha
        acc5 *= alpha
        acc6 *= alpha
        acc7 *= alpha

        m_for_p = tl.where(has_any, m_new, 0.0)
        p = tl.exp(logits - m_for_p)
        p = tl.where(valid, p, 0.0)
        l += tl.sum(p, axis=0)

        d0 = 0 * BD + offs_bd
        v = tl.load(
            ckv_ptr + row_ckv[:, None] + d0[None, :].to(tl.int64) * sckvd,
            mask=valid[:, None],
            other=0,
            cache_modifier=".cg",
            eviction_policy="evict_first",
        ).to(tl.float32)
        acc0 += tl.sum(v * p[:, None], axis=0)

        d0 = 1 * BD + offs_bd
        v = tl.load(
            ckv_ptr + row_ckv[:, None] + d0[None, :].to(tl.int64) * sckvd,
            mask=valid[:, None],
            other=0,
            cache_modifier=".cg",
            eviction_policy="evict_first",
        ).to(tl.float32)
        acc1 += tl.sum(v * p[:, None], axis=0)

        d0 = 2 * BD + offs_bd
        v = tl.load(
            ckv_ptr + row_ckv[:, None] + d0[None, :].to(tl.int64) * sckvd,
            mask=valid[:, None],
            other=0,
            cache_modifier=".cg",
            eviction_policy="evict_first",
        ).to(tl.float32)
        acc2 += tl.sum(v * p[:, None], axis=0)

        d0 = 3 * BD + offs_bd
        v = tl.load(
            ckv_ptr + row_ckv[:, None] + d0[None, :].to(tl.int64) * sckvd,
            mask=valid[:, None],
            other=0,
            cache_modifier=".cg",
            eviction_policy="evict_first",
        ).to(tl.float32)
        acc3 += tl.sum(v * p[:, None], axis=0)

        d0 = 4 * BD + offs_bd
        v = tl.load(
            ckv_ptr + row_ckv[:, None] + d0[None, :].to(tl.int64) * sckvd,
            mask=valid[:, None],
            other=0,
            cache_modifier=".cg",
            eviction_policy="evict_first",
        ).to(tl.float32)
        acc4 += tl.sum(v * p[:, None], axis=0)

        d0 = 5 * BD + offs_bd
        v = tl.load(
            ckv_ptr + row_ckv[:, None] + d0[None, :].to(tl.int64) * sckvd,
            mask=valid[:, None],
            other=0,
            cache_modifier=".cg",
            eviction_policy="evict_first",
        ).to(tl.float32)
        acc5 += tl.sum(v * p[:, None], axis=0)

        d0 = 6 * BD + offs_bd
        v = tl.load(
            ckv_ptr + row_ckv[:, None] + d0[None, :].to(tl.int64) * sckvd,
            mask=valid[:, None],
            other=0,
            cache_modifier=".cg",
            eviction_policy="evict_first",
        ).to(tl.float32)
        acc6 += tl.sum(v * p[:, None], axis=0)

        d0 = 7 * BD + offs_bd
        v = tl.load(
            ckv_ptr + row_ckv[:, None] + d0[None, :].to(tl.int64) * sckvd,
            mask=valid[:, None],
            other=0,
            cache_modifier=".cg",
            eviction_policy="evict_first",
        ).to(tl.float32)
        acc7 += tl.sum(v * p[:, None], axis=0)

        m = m_new

    inv_l = tl.where(l > 0.0, 1.0 / l, 0.0)
    mask_out = inb & (offs_bd < D_CKV_L)

    outv = tl.where(l > 0.0, acc0 * inv_l, 0.0)
    tl.store(out_ptr + t * s_out_t + h * s_out_h + (0 * BD + offs_bd) * s_out_d, outv.to(OUT_DTYPE), mask=mask_out)
    outv = tl.where(l > 0.0, acc1 * inv_l, 0.0)
    tl.store(out_ptr + t * s_out_t + h * s_out_h + (1 * BD + offs_bd) * s_out_d, outv.to(OUT_DTYPE), mask=mask_out)
    outv = tl.where(l > 0.0, acc2 * inv_l, 0.0)
    tl.store(out_ptr + t * s_out_t + h * s_out_h + (2 * BD + offs_bd) * s_out_d, outv.to(OUT_DTYPE), mask=mask_out)
    outv = tl.where(l > 0.0, acc3 * inv_l, 0.0)
    tl.store(out_ptr + t * s_out_t + h * s_out_h + (3 * BD + offs_bd) * s_out_d, outv.to(OUT_DTYPE), mask=mask_out)
    outv = tl.where(l > 0.0, acc4 * inv_l, 0.0)
    tl.store(out_ptr + t * s_out_t + h * s_out_h + (4 * BD + offs_bd) * s_out_d, outv.to(OUT_DTYPE), mask=mask_out)
    outv = tl.where(l > 0.0, acc5 * inv_l, 0.0)
    tl.store(out_ptr + t * s_out_t + h * s_out_h + (5 * BD + offs_bd) * s_out_d, outv.to(OUT_DTYPE), mask=mask_out)
    outv = tl.where(l > 0.0, acc6 * inv_l, 0.0)
    tl.store(out_ptr + t * s_out_t + h * s_out_h + (6 * BD + offs_bd) * s_out_d, outv.to(OUT_DTYPE), mask=mask_out)
    outv = tl.where(l > 0.0, acc7 * inv_l, 0.0)
    tl.store(out_ptr + t * s_out_t + h * s_out_h + (7 * BD + offs_bd) * s_out_d, outv.to(OUT_DTYPE), mask=mask_out)

    if HAS_LSE:
        lse_val = tl.where(l > 0.0, (m + tl.log(l)) * INV_LN2_L, -float("inf"))
        tl.store(lse_ptr + t * s_lse_t + h * s_lse_h, lse_val, mask=inb)


def _as_float32_scalar_tensor(sm_scale: torch.Tensor | float, device: torch.device) -> torch.Tensor:
    if isinstance(sm_scale, (float, int)):
        return torch.tensor([float(sm_scale)], device=device, dtype=torch.float32)
    if not torch.is_tensor(sm_scale):
        raise TypeError("sm_scale must be a float or a torch.Tensor")
    if sm_scale.numel() != 1:
        raise ValueError("sm_scale must have numel()==1")
    if sm_scale.device != device:
        sm_scale = sm_scale.to(device)
    if sm_scale.dtype != torch.float32:
        sm_scale = sm_scale.float()
    return sm_scale.reshape(1)


def kernel(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    ckv_cache: torch.Tensor,
    kpe_cache: torch.Tensor,
    sparse_indices: torch.Tensor,
    sm_scale: torch.Tensor | float,
    output: torch.Tensor,
    lse: torch.Tensor,
):
    if not (
        q_nope.is_cuda
        and q_pe.is_cuda
        and ckv_cache.is_cuda
        and kpe_cache.is_cuda
        and sparse_indices.is_cuda
        and output.is_cuda
        and lse.is_cuda
    ):
        raise ValueError("All inputs and outputs must be CUDA tensors.")

    T, H, D = q_nope.shape
    if H != NUM_QO_HEADS or D != HEAD_DIM_CKV:
        raise ValueError(f"q_nope must be [T,16,512], got {tuple(q_nope.shape)}")
    if q_pe.shape != (T, NUM_QO_HEADS, HEAD_DIM_KPE):
        raise ValueError(f"q_pe must be [T,16,64], got {tuple(q_pe.shape)}")

    P, PS, DC = ckv_cache.shape
    if (PS != PAGE_SIZE) or (DC != HEAD_DIM_CKV):
        raise ValueError(f"ckv_cache must be [P,64,512], got {tuple(ckv_cache.shape)}")
    if kpe_cache.shape != (P, PAGE_SIZE, HEAD_DIM_KPE):
        raise ValueError(f"kpe_cache must be [P,64,64], got {tuple(kpe_cache.shape)}")

    if sparse_indices.shape != (T, TOPK):
        raise ValueError(f"sparse_indices must be [T,2048], got {tuple(sparse_indices.shape)}")
    if sparse_indices.dtype != torch.int32:
        raise ValueError(f"sparse_indices must be int32, got {sparse_indices.dtype}")

    if output.shape != (T, NUM_QO_HEADS, HEAD_DIM_CKV):
        raise ValueError(f"output must be [T,16,512], got {tuple(output.shape)}")
    if output.dtype != torch.bfloat16:
        raise ValueError(f"output must be bfloat16, got {output.dtype}")

    if lse.shape != (T, NUM_QO_HEADS):
        raise ValueError(f"lse must be [T,16], got {tuple(lse.shape)}")
    if lse.dtype != torch.float32:
        raise ValueError(f"lse must be float32, got {lse.dtype}")

    kv_capacity = P * PAGE_SIZE
    sm_scale_t = _as_float32_scalar_tensor(sm_scale, device=q_nope.device)

    grid = (T * NUM_QO_HEADS,)

    _kernel[grid](
        q_nope,
        q_pe,
        ckv_cache,
        kpe_cache,
        sparse_indices,
        sm_scale_t,
        output,
        lse,
        num_tokens=T,
        kv_capacity=kv_capacity,
        num_heads=NUM_QO_HEADS,
        s_qn_t=q_nope.stride(0),
        s_qn_h=q_nope.stride(1),
        s_qn_d=q_nope.stride(2),
        s_qp_t=q_pe.stride(0),
        s_qp_h=q_pe.stride(1),
        s_qp_d=q_pe.stride(2),
        s_ckv_p=ckv_cache.stride(0),
        s_ckv_s=ckv_cache.stride(1),
        s_ckv_d=ckv_cache.stride(2),
        s_kpe_p=kpe_cache.stride(0),
        s_kpe_s=kpe_cache.stride(1),
        s_kpe_d=kpe_cache.stride(2),
        s_si_t=sparse_indices.stride(0),
        s_si_k=sparse_indices.stride(1),
        s_out_t=output.stride(0),
        s_out_h=output.stride(1),
        s_out_d=output.stride(2),
        s_lse_t=lse.stride(0),
        s_lse_h=lse.stride(1),
        HAS_LSE=1,
        OUT_DTYPE=tl.bfloat16,
        BLOCK_K=BLOCK_K_FIXED,
        BD=BD_FIXED,
        BDP=BDP_FIXED,
        LOOP_STAGES=LOOP_STAGES_FIXED,
        num_warps=NUM_WARPS_FIXED,
        num_stages=NUM_STAGES_FIXED,
    )