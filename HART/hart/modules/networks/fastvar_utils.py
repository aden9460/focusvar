import torch
from typing import Callable, Dict, Tuple


FASTVAR_COMPUTE_MERGE_ENABLED = False
SPACEVAR_ENABLED = False
FASTVAR_VERBOSE = True
FASTVAR_RATIO_BY_SCALE: Dict[int, float] = {48: 0.4, 64: 0.5}
FASTVAR_START_LAYER = 0
FASTVAR_PRUNE_SCALES = {48, 64}


def set_fastvar_verbose(enabled: bool):
    global FASTVAR_VERBOSE
    FASTVAR_VERBOSE = bool(enabled)


def set_fastvar_compute_merge_enabled(enabled: bool):
    global FASTVAR_COMPUTE_MERGE_ENABLED
    FASTVAR_COMPUTE_MERGE_ENABLED = bool(enabled)


def set_spacevar_config(enabled: bool, ratio_by_scale=None):
    global SPACEVAR_ENABLED, FASTVAR_RATIO_BY_SCALE
    SPACEVAR_ENABLED = bool(enabled)
    if ratio_by_scale is not None:
        FASTVAR_RATIO_BY_SCALE = {int(k): float(v) for k, v in ratio_by_scale.items()}


def set_fastvar_prune_scales(scales):
    global FASTVAR_PRUNE_SCALES
    FASTVAR_PRUNE_SCALES = {int(s) for s in scales}


def set_fastvar_start_layer(layer: int):
    global FASTVAR_START_LAYER
    FASTVAR_START_LAYER = int(layer)


def do_nothing(x: torch.Tensor, *args, **kwargs):
    return x


def build_merge_functions_from_indices(select_indices: torch.Tensor, cur_shape):
    if select_indices is None or cur_shape is None:
        return do_nothing, do_nothing, do_nothing

    select_indices = select_indices.to(dtype=torch.long)
    if select_indices.ndim != 3 or select_indices.shape[-1] != 1:
        raise ValueError(f"select_indices shape must be [B, K, 1], got {tuple(select_indices.shape)}")

    _, H, W = cur_shape
    keep_B, _, _ = select_indices.shape
    last_full_x = None

    def merge(merged_cur_x):
        nonlocal last_full_x
        c = merged_cur_x.shape[-1]
        last_full_x = merged_cur_x
        return torch.gather(merged_cur_x, dim=1, index=select_indices.repeat(1, 1, c))

    def unmerge(unmerged_cur_x, unmerged_cache_x, cached_hw=None):
        c = unmerged_cur_x.shape[-1]
        L = H * W
        if unmerged_cache_x is None:
            if last_full_x is None:
                raise ValueError("full-token fallback cache is unavailable for selected indices")
            base_x = last_full_x.clone()
        else:
            if cached_hw is None:
                cached_hw = (int(unmerged_cache_x.shape[1] ** 0.5), int(unmerged_cache_x.shape[1] ** 0.5))
            base_x = unmerged_cache_x.view(keep_B, cached_hw[0], cached_hw[1], -1).permute(0, 3, 1, 2)
            base_x = torch.nn.functional.interpolate(base_x, size=(H, W), mode="area").permute(0, 2, 3, 1).view(keep_B, L, c)
        base_x = base_x.to(device=unmerged_cur_x.device, dtype=unmerged_cur_x.dtype)
        base_x.scatter_(dim=1, index=select_indices.repeat(1, 1, c), src=unmerged_cur_x)
        return base_x

    def get_src_tgt_idx():
        return select_indices

    return merge, unmerge, get_src_tgt_idx


def build_cfg_diff_select_indices(cur_x, num_remain):
    B, _, _ = cur_x.shape
    if B <= 1 or B % 2 != 0:
        raise ValueError(f"cfg_diff select indices require dual-branch batch, got B={B}")
    real_B = B // 2
    x_cond = cur_x[:real_B]
    x_uncond = cur_x[real_B:]
    cfg_diff = torch.norm(x_cond - x_uncond, p=2, dim=-1, keepdim=True)
    select_indices = torch.argsort(cfg_diff, dim=1, descending=True)[:, :num_remain, :]
    return select_indices.repeat(2, 1, 1)


def masked_previous_scale_cache(cur_x, num_remain, cur_shape):
    B, _, c = cur_x.shape
    _, H, W = cur_shape
    mean_x = cur_x.view(B, H, W, -1).permute(0, 3, 1, 2)
    mean_x = torch.nn.functional.adaptive_avg_pool2d(mean_x, (1, 1)).permute(0, 2, 3, 1).view(B, 1, c)
    score = torch.sum((cur_x - mean_x) ** 2, dim=-1, keepdim=True)
    select_indices = torch.argsort(score, dim=1, descending=True)[:, :num_remain, :]
    return build_merge_functions_from_indices(select_indices, cur_shape)


def semantic_masked_previous_scale_cache(cur_x, num_remain, cur_shape):
    select_indices = build_cfg_diff_select_indices(cur_x, num_remain)
    return build_merge_functions_from_indices(select_indices, cur_shape)


def compute_merge(
    x: torch.Tensor,
    prune_scale_list=None,
    is_later_layer=False,
    x_shape=None,
    layer_idx: int = None,
    external_select_indices=None,
) -> Tuple[Callable, ...]:
    if x_shape is None:
        return do_nothing, do_nothing, do_nothing

    if external_select_indices is not None:
        return build_merge_functions_from_indices(external_select_indices, x_shape)

    _, H, W = x_shape
    if H != W or H * W != x.shape[1]:
        return do_nothing, do_nothing, do_nothing

    active_scales = FASTVAR_PRUNE_SCALES if prune_scale_list is None else {int(s) for s in prune_scale_list}
    if W not in active_scales or not is_later_layer:
        return do_nothing, do_nothing, do_nothing

    if layer_idx is not None and int(layer_idx) < FASTVAR_START_LAYER:
        return do_nothing, do_nothing, do_nothing

    keep_ratio = FASTVAR_RATIO_BY_SCALE.get(W)
    if keep_ratio is None or keep_ratio <= 0:
        return do_nothing, do_nothing, do_nothing

    num_remain = max(1, int(x.shape[1] * keep_ratio))
    if num_remain >= x.shape[1]:
        return do_nothing, do_nothing, do_nothing

    use_fastvar = FASTVAR_COMPUTE_MERGE_ENABLED
    use_spacevar = SPACEVAR_ENABLED and x.shape[0] > 1 and x.shape[0] % 2 == 0
    if not use_fastvar and not use_spacevar:
        return do_nothing, do_nothing, do_nothing

    if use_fastvar:
        if FASTVAR_VERBOSE:
            print(f"[hart-fastvar] scale={H}x{W} layer={layer_idx} keep_ratio={keep_ratio:.3f} keep={num_remain}/{x.shape[1]}")
        return masked_previous_scale_cache(x, num_remain, x_shape)

    if FASTVAR_VERBOSE:
        print(f"[hart-spacevar] scale={H}x{W} layer={layer_idx} keep_ratio={keep_ratio:.3f} keep={num_remain}/{x.shape[1]}")
    return semantic_masked_previous_scale_cache(x, num_remain, x_shape)
