"""
Batched single-device Muon (with auxiliary Adam).

Same update as muon.SingleDeviceMuonWithAuxAdam (Keller Jordan's reference implementation), but the per-parameter
work is batched: momentum, weight decay and the parameter update run as torch._foreach ops, and the Newton-Schulz
orthogonalization runs once per group of same-shaped matrices instead of once per matrix. A LoRA consists of hundreds
of small matrices that share only a handful of shapes, so the reference implementation's per-parameter Python loop
(15 tiny matmuls per matrix) dominates its step time. Idea from Fizgig's BatchedMuon (Apache-2.0).

State dict keys are the same as the reference implementation, so a backup made with it can be resumed.
"""
import math

import torch
from torch import Tensor

_NS_A, _NS_B, _NS_C = 3.4445, -4.7750, 2.0315

# upper bound on the number of elements stacked into one batched Newton-Schulz call, to bound the temporary memory
# when many large matrices share a shape (e.g. a full fine-tune)
_MAX_BUCKET_NUMEL = 64 * 1024 * 1024


def zeropower_via_newtonschulz5_batched(G: Tensor, steps: int) -> Tensor:
    """muon.zeropower_via_newtonschulz5 for a stack of same-shaped matrices (B, m, n)."""
    X = G.bfloat16()
    tall = G.size(-2) > G.size(-1)
    if tall:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = torch.baddbmm(A, A, A, beta=_NS_B, alpha=_NS_C)  # b * A + c * A @ A
        X = torch.baddbmm(X, B, X, beta=_NS_A)  # a * X + B @ X
    if tall:
        X = X.mT
    return X


def _lr_scale(shape: tuple[int, ...], match_rms_adamw: bool) -> float:
    if match_rms_adamw:
        # update RMS ~= 0.2, like AdamW's, so AdamW learning rates carry over (Moonlight / torch.optim.Muon's
        # adjust_lr_fn="match_rms_adamw"). Uses the orthogonalized 2D matrix's shape.
        rows = shape[0]
        cols = math.prod(shape[1:])
        return 0.2 * math.sqrt(max(rows, cols))
    # the reference implementation uses the last two dimensions of the gradient, also for 4D conv weights
    return max(1, shape[-2] / shape[-1]) ** 0.5


class BatchedSingleDeviceMuonWithAuxAdam(torch.optim.Optimizer):
    """
    Drop-in replacement for muon.SingleDeviceMuonWithAuxAdam. Takes the same param groups ("use_muon" decides the
    update rule), plus these optional Muon group keys:
      - nesterov (default True)
      - ns_steps (default 5)
      - match_rms_adamw (default False): scale updates to AdamW's RMS instead of the reference sqrt(max(1, A/B))
    Extra keys (e.g. initial_lr, name, optim_type) are allowed. The optional keys are read with defaults, because
    loading a reference implementation's state dict replaces the param groups with ones that don't have them.
    """

    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                group.setdefault("lr", 0.02)
                group.setdefault("momentum", 0.95)
                group.setdefault("weight_decay", 0)
                group.setdefault("nesterov", True)
                group.setdefault("ns_steps", 5)
                group.setdefault("match_rms_adamw", False)
                for p in group["params"]:
                    if p.ndim < 2:
                        raise ValueError(f"Muon only supports parameters with 2 or more dimensions, got {tuple(p.shape)}. "
                                         "Enable MuonWithAuxAdam to train them with Adam.")
            else:
                group.setdefault("lr", 3e-4)
                group.setdefault("betas", (0.9, 0.95))
                group.setdefault("eps", 1e-10)
                group.setdefault("weight_decay", 0)
        super().__init__(param_groups, {})

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if not group["params"]:
                continue
            for p in group["params"]:
                if p.grad is None:
                    p.grad = torch.zeros_like(p)  # same as the reference implementation
            if group["use_muon"]:
                self._muon_group_step(group)
            else:
                self._adam_group_step(group)

        return loss

    def _muon_group_step(self, group: dict):
        params = group["params"]
        lr = group["lr"]
        beta = group["momentum"]
        grads = [p.grad for p in params]

        bufs = []
        for p in params:
            state = self.state[p]
            if len(state) == 0:
                state["momentum_buffer"] = torch.zeros_like(p)
            bufs.append(state["momentum_buffer"])

        torch._foreach_lerp_(bufs, grads, 1 - beta)
        updates = torch._foreach_lerp(grads, bufs, beta) if group.get("nesterov", True) else bufs

        if group["weight_decay"] != 0:
            torch._foreach_mul_(params, 1 - lr * group["weight_decay"])

        buckets: dict[tuple, list[int]] = {}
        for i, p in enumerate(params):
            buckets.setdefault((tuple(p.shape), p.device, p.dtype), []).append(i)

        for (shape, _device, _dtype), indices in buckets.items():
            numel = math.prod(shape)
            chunk_size = max(1, _MAX_BUCKET_NUMEL // numel)
            scale = _lr_scale(shape, group.get("match_rms_adamw", False))
            for start in range(0, len(indices), chunk_size):
                chunk = indices[start:start + chunk_size]
                stacked = torch.stack([updates[i].reshape(shape[0], -1) for i in chunk])
                ortho = zeropower_via_newtonschulz5_batched(stacked, group.get("ns_steps", 5))
                ortho *= scale  # in bf16, like the reference implementation
                chunk_params = [params[i] for i in chunk]
                ortho = ortho.to(chunk_params[0].dtype)
                torch._foreach_add_(chunk_params, [o.view(shape) for o in ortho.unbind(0)], alpha=-lr)

    def _adam_group_step(self, group: dict):
        params = group["params"]
        beta1, beta2 = group["betas"]
        grads = [p.grad for p in params]

        exp_avgs, exp_avg_sqs, bias1, bias2 = [], [], [], []
        for p in params:
            state = self.state[p]
            if len(state) == 0:
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)
                state["step"] = 0
            state["step"] += 1
            exp_avgs.append(state["exp_avg"])
            exp_avg_sqs.append(state["exp_avg_sq"])
            bias1.append(1 - beta1 ** state["step"])
            bias2.append(1 - beta2 ** state["step"])

        torch._foreach_lerp_(exp_avgs, grads, 1 - beta1)
        torch._foreach_lerp_(exp_avg_sqs, torch._foreach_mul(grads, grads), 1 - beta2)
        denom = torch._foreach_div(exp_avg_sqs, bias2)
        torch._foreach_sqrt_(denom)
        torch._foreach_add_(denom, group["eps"])
        update = torch._foreach_div(exp_avgs, bias1)
        torch._foreach_div_(update, denom)

        if group["weight_decay"] != 0:
            torch._foreach_mul_(params, 1 - group["lr"] * group["weight_decay"])
        torch._foreach_add_(params, update, alpha=-group["lr"])
