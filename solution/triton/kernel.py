import math

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _gdn_decode_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    state_ptr,
    new_state_ptr,
    gate_ptr,
    beta_ptr,
    scale,
    out_ptr,
    stride_qb,
    stride_qh,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vd,
    stride_sb,
    stride_sh,
    stride_sd,
    stride_sk,
    stride_ob,
    stride_oh,
    stride_od,
    stride_gb,
    stride_gh,
    HEAD_DIM: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    GVA_RATIO: tl.constexpr,
):
    b_idx = tl.program_id(0)
    vh_idx = tl.program_id(1)
    tile_idx = tl.program_id(2)
    qh_idx = vh_idx // GVA_RATIO

    gate = tl.load(gate_ptr + b_idx * stride_gb + vh_idx * stride_gh)
    beta = tl.load(beta_ptr + b_idx * stride_gb + vh_idx * stride_gh)

    kd = tl.arange(0, HEAD_DIM)
    k_vec = (
        tl.load(k_ptr + b_idx * stride_kb + qh_idx * stride_kh + kd * stride_kd).to(
            tl.float32
        )
        * scale
    )
    q_vec = tl.load(q_ptr + b_idx * stride_qb + qh_idx * stride_qh + kd * stride_qd).to(
        tl.float32
    )

    rows = tile_idx * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    v_tile = tl.load(
        v_ptr + b_idx * stride_vb + vh_idx * stride_vh + rows * stride_vd
    ).to(tl.float32)

    s_base = state_ptr + b_idx * stride_sb + vh_idx * stride_sh
    state_tile = tl.load(
        s_base + rows[:, None] * stride_sd + kd[None, :] * stride_sk
    ).to(tl.float32)

    retrieve = tl.sum(state_tile * k_vec[None, :], axis=1)
    delta = v_tile - retrieve
    new_state = gate * state_tile + beta * delta[:, None] * k_vec[None, :]
    out_vals = tl.sum(new_state * q_vec[None, :], axis=1)

    ns_base = new_state_ptr + b_idx * stride_sb + vh_idx * stride_sh
    tl.store(
        ns_base + rows[:, None] * stride_sd + kd[None, :] * stride_sk,
        new_state.to(tl.float32),
    )

    o_base = out_ptr + b_idx * stride_ob + vh_idx * stride_oh
    tl.store(o_base + rows * stride_od, out_vals.to(tl.bfloat16))


def kernel(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    state: torch.Tensor | None,
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    b: torch.Tensor,
    scale,
    output: torch.Tensor,
    new_state: torch.Tensor,
):
    B, T, Hq, D = q.shape
    Hv = v.shape[2]

    if state is None:
        state = torch.zeros_like(new_state)

    sc = (
        float(scale.item())
        if isinstance(scale, torch.Tensor)
        else (1.0 / math.sqrt(D) if not scale else float(scale))
    )

    # Mamba2 discretization: gate = exp(-exp(A_log) * softplus(a + dt_bias))
    dt = F.softplus(a.float().squeeze(1) + dt_bias.float()[None, :])
    gate = torch.exp(-torch.exp(A_log.float())[None, :] * dt).contiguous()
    beta = torch.sigmoid(b.float().squeeze(1)).contiguous()

    GVA_RATIO = Hv // Hq  # 2

    total = B * Hv
    if total >= 592:
        BLOCK_ROWS, NW, NS = 128, 4, 3
    elif total >= 296:
        BLOCK_ROWS, NW, NS = 64, 2, 3
    elif total >= 148:
        BLOCK_ROWS, NW, NS = 32, 1, 4
    else:
        BLOCK_ROWS, NW, NS = 16, 1, 4

    grid = (B, Hv, D // BLOCK_ROWS)

    _gdn_decode_kernel[grid](
        q,
        k,
        v,
        state,
        new_state,
        gate,
        beta,
        sc,
        output,
        q.stride(0),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(2),
        v.stride(3),
        state.stride(0),
        state.stride(1),
        state.stride(2),
        state.stride(3),
        output.stride(0),
        output.stride(2),
        output.stride(3),
        gate.stride(0),
        gate.stride(1),
        HEAD_DIM=128,
        BLOCK_ROWS=BLOCK_ROWS,
        GVA_RATIO=GVA_RATIO,
        num_warps=NW,
        num_stages=NS,
    )
