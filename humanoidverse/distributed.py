from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from typing import Any

import torch
import torch.distributed as dist


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def rank() -> int:
    return dist.get_rank() if is_distributed() else int(os.environ.get("RANK", "0"))


def world_size() -> int:
    return dist.get_world_size() if is_distributed() else int(os.environ.get("WORLD_SIZE", "1"))


def barrier() -> None:
    if is_distributed():
        if torch.cuda.is_available():
            dist.barrier(device_ids=[torch.cuda.current_device()])
        else:
            dist.barrier()


@torch.compiler.disable
def average_gradients(parameters: Iterable[torch.nn.Parameter]) -> None:
    if not is_distributed():
        return
    grads = [param.grad for param in parameters if param.grad is not None]
    if not grads:
        return
    # One allreduce per dtype, in chunks. Same sum-and-average as per-tensor NCCL.
    scale = 1.0 / float(dist.get_world_size())
    buckets: dict[tuple[torch.dtype, torch.device], list[torch.Tensor]] = {}
    for grad in grads:
        buckets.setdefault((grad.dtype, grad.device), []).append(grad)
    chunk_elems = 32_000_000
    for group in buckets.values():
        flat = torch.cat([grad.detach().reshape(-1) for grad in group])
        for start in range(0, flat.numel(), chunk_elems):
            piece = flat[start : start + chunk_elems]
            dist.all_reduce(piece, op=dist.ReduceOp.SUM)
        flat.mul_(scale)
        offset = 0
        for grad in group:
            count = grad.numel()
            grad.copy_(flat[offset : offset + count].view_as(grad))
            offset += count


@torch.compiler.disable
def broadcast_module_state(module: torch.nn.Module, src: int = 0, *, obs_freeze_guard=None, coordinate=None) -> None:
    if obs_freeze_guard is None:
        if not is_distributed():
            return
        for tensor in list(module.parameters()) + list(module.buffers()):
            dist.broadcast(tensor.data, src=src)
        return
    from humanoidverse.training.obs_normalizer_sync import require_guard
    require_guard(obs_freeze_guard)
    obs_freeze_guard.check("initial_broadcast_before", coordinate, module)
    protected_ids = obs_freeze_guard.protected_ids(module, "initial_broadcast_before")
    if is_distributed():
        for tensor in list(module.parameters()) + list(module.buffers()):
            if id(tensor) not in protected_ids:
                dist.broadcast(tensor.data, src=src)
    obs_freeze_guard.check("initial_broadcast_after", coordinate, module)


@torch.compiler.disable
def sync_floating_buffers(module: torch.nn.Module, src: int = 0, *, obs_freeze_guard=None, coordinate=None) -> None:
    if obs_freeze_guard is not None:
        from humanoidverse.training.obs_normalizer_sync import require_guard
        require_guard(obs_freeze_guard)
        obs_freeze_guard.check("sync_before", coordinate, module)
        protected_ids = obs_freeze_guard.protected_ids(module, "sync_before")
    else:
        protected_ids = frozenset()
    if not is_distributed():
        if obs_freeze_guard is not None:
            obs_freeze_guard.check("sync_after", coordinate, module)
        return
    floating = [
        buffer
        for buffer in module.buffers()
        if id(buffer) not in protected_ids and (buffer.dtype.is_floating_point or buffer.dtype.is_complex)
    ]
    if floating:
        scale = 1.0 / float(dist.get_world_size())
        buckets: dict[tuple[torch.dtype, torch.device], list[torch.Tensor]] = {}
        for buffer in floating:
            buckets.setdefault((buffer.dtype, buffer.device), []).append(buffer)
        chunk_elems = 32_000_000
        for group in buckets.values():
            flat = torch.cat([buffer.detach().reshape(-1) for buffer in group])
            for start in range(0, flat.numel(), chunk_elems):
                piece = flat[start : start + chunk_elems]
                dist.all_reduce(piece, op=dist.ReduceOp.SUM)
            flat.mul_(scale)
            offset = 0
            for buffer in group:
                count = buffer.numel()
                buffer.copy_(flat[offset : offset + count].view_as(buffer))
                offset += count
    for buffer in module.buffers():
        if id(buffer) not in protected_ids and not (buffer.dtype.is_floating_point or buffer.dtype.is_complex):
            dist.broadcast(buffer.data, src=src)
    if obs_freeze_guard is not None:
        obs_freeze_guard.check("sync_after", coordinate, module)


@torch.compiler.disable
def broadcast_optimizer_state(optimizer: torch.optim.Optimizer, src: int = 0) -> None:
    if not is_distributed():
        return
    for state in optimizer.state.values():
        for value in state.values():
            if torch.is_tensor(value):
                dist.broadcast(value.data, src=src)


def agent_optimizers(agent: Any) -> list[torch.optim.Optimizer]:
    names = [
        "actor_optimizer",
        "backward_optimizer",
        "forward_optimizer",
        "critic_optimizer",
        "discriminator_optimizer",
        "aux_critic_optimizer",
    ]
    return [getattr(agent, name) for name in names if hasattr(agent, name)]


def broadcast_agent_state(agent: Any, src: int = 0, *, obs_freeze_guard=None, coordinate=None) -> None:
    if not is_distributed() and obs_freeze_guard is None:
        return
    broadcast_module_state(agent._model, src=src, obs_freeze_guard=obs_freeze_guard, coordinate=coordinate)
    for optimizer in agent_optimizers(agent):
        broadcast_optimizer_state(optimizer, src=src)


@torch.compiler.disable
def broadcast_object(value: Any, src: int = 0) -> Any:
    if not is_distributed():
        return value
    objects = [value if dist.get_rank() == src else None]
    dist.broadcast_object_list(objects, src=src)
    return objects[0]


@torch.compiler.disable
def module_sync_report(module: torch.nn.Module, src: int = 0) -> dict[str, Any]:
    if not is_distributed():
        return {"world_size": 1, "rank": 0, "max_abs_diff_from_rank0": 0.0}

    tensors = list(module.parameters()) + [
        buffer for buffer in module.buffers() if buffer.dtype.is_floating_point or buffer.dtype.is_complex
    ]
    if tensors:
        device = tensors[0].device
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    local_summary = torch.zeros(3, dtype=torch.float64, device=device)
    max_abs_diff = torch.zeros((), dtype=torch.float32, device=device)

    for tensor in tensors:
        detached = tensor.detach()
        local_summary[0] += detached.double().sum()
        local_summary[1] += detached.double().square().sum()
        local_summary[2] += detached.numel()
        reference = detached.clone()
        dist.broadcast(reference, src=src)
        if detached.numel() > 0:
            diff = (detached - reference).abs().max().float()
            max_abs_diff = torch.maximum(max_abs_diff, diff)

    dist.all_reduce(max_abs_diff, op=dist.ReduceOp.MAX)
    gathered = [torch.empty_like(local_summary) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, local_summary)
    summaries = torch.stack(gathered).cpu()
    return {
        "world_size": dist.get_world_size(),
        "rank": dist.get_rank(),
        "max_abs_diff_from_rank0": float(max_abs_diff.cpu().item()),
        "rank_param_buffer_sum": [float(x) for x in summaries[:, 0].tolist()],
        "rank_param_buffer_sqsum": [float(x) for x in summaries[:, 1].tolist()],
        "rank_param_buffer_numel": [int(x) for x in summaries[:, 2].tolist()],
    }


@torch.compiler.disable
def average_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    if not is_distributed():
        return dict(metrics)
    reduced: dict[str, Any] = {}
    scale = 1.0 / float(dist.get_world_size())
    for key, value in metrics.items():
        if torch.is_tensor(value):
            tensor = value.detach().clone()
            if not (tensor.dtype.is_floating_point or tensor.dtype.is_complex):
                tensor = tensor.float()
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            reduced[key] = tensor * scale
        else:
            reduced[key] = value
    return reduced
