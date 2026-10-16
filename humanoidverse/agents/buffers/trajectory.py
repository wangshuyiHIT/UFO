import dataclasses
import typing
from collections.abc import Mapping
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from tensordict import TensorDict
from torch.utils._pytree import tree_map

from .transition import DictBuffer, _to_torch

INT_CLASSES_TYPING = Union[int, np.integer]
INT_CLASSES = typing.get_args(INT_CLASSES_TYPING)


def _is_int(index):
    if isinstance(index, INT_CLASSES):
        return True
    if isinstance(index, (np.ndarray, torch.Tensor)):
        return index.ndim == 0
    return False


def tree_concat(list_of_pytree_of_tensors, dim=0):
    """Slow-ish implementation of concatenating leaves of a pytree with matching structures"""
    tds = tuple(map(lambda x: TensorDict.from_pytree(x, auto_batch_size=True), list_of_pytree_of_tensors))
    concatenated = torch.cat(tds, dim=dim)
    # Return a non-tensordict object to stay consistent with rest of the code
    if isinstance(concatenated, TensorDict):
        return concatenated.to_pytree()
    # but if input was not list of pytrees, the output is already a tensor
    return concatenated


class TrajectoryDictBuffer:
    # TODO(team): maybe pass to TENSORDICT to have better support for
    # several operations
    def __init__(
        self,
        episodes: List[dict],
        device: str = "cpu",
        seq_length: int = 1,
        output_key_t: List[str] = ["observation"],
        output_key_tp1: List[str] = ["observation"],
        end_key: Tuple[str] | str = "done",
        motion_id_key: Tuple[str] | str = "motion_id",
        trajectory_source_labels: List[str] | None = None,
        source_weights: Dict[str, float] | None = None,
        payload_device: str | None = None,
    ) -> None:
        self._is_full = True
        self.output_key_t = output_key_t
        self.output_key_tp1 = output_key_tp1
        self.seq_length = seq_length
        self.device = device
        if payload_device not in (None, "cpu"):
            raise ValueError("Only the explicit CPU payload residency mode is supported")
        self.payload_device = payload_device
        storage_device = device if payload_device is None else payload_device
        self.end_key = end_key
        self.motion_id_key = motion_id_key

        def add_new_data(data, storage):
            for k, v in data.items():
                if isinstance(v, Mapping):
                    # If the value is a dictionary, recursively call the function
                    if k not in storage:
                        storage[k] = {}
                    add_new_data(v, storage=storage[k])
                else:
                    if k not in storage:
                        storage[k] = []
                    storage[k].append(_to_torch(v, device=storage_device))

        def concat_dict(storage):
            for k, v in storage.items():
                if isinstance(v, Mapping):
                    concat_dict(storage[k])
                else:
                    storage[k] = torch.cat(v, axis=0)

        self.storage = {}
        set_key(self.storage, self.end_key, [])
        self.motion_ids = []
        for ep in episodes:
            _k = [k for k, v in ep.items() if not isinstance(v, Mapping)][0]
            # first append all the values to a list
            add_new_data(ep, self.storage)
            # add termination of each episode
            if not key_exists(ep, self.end_key):
                episode_start = torch.zeros((ep[_k].shape[0], 1), dtype=torch.bool, device=storage_device)
                episode_start[-1] = True
                get_key(self.storage, self.end_key).append(episode_start)
            self.motion_ids.append(get_key(ep, self.motion_id_key)[0].item())
        # then concatenate them into a single tensor
        concat_dict(self.storage)
        assert get_key(self.storage, self.end_key).dtype == torch.bool
        done = get_key(self.storage, self.end_key)
        if self.payload_device is not None:
            done = done.to(self.device)
        # cursor is none since we don't wrap data
        self.start_idx, self.stop_idx, self.lengths = find_start_stop_traj(
            done.squeeze()[: len(self)], at_capacity=self._is_full, cursor=None
        )
        # set priorities to match the number of trajectories
        self.priorities = torch.ones(len(self.lengths), device=self.device, dtype=torch.float32) / len(self.lengths)
        self.source_weights = source_weights
        self.trajectory_source_ids = None
        if (trajectory_source_labels is None) != (source_weights is None):
            raise ValueError("Expert source labels and weights must be supplied together")
        if source_weights is not None:
            if (len(trajectory_source_labels) != len(self.lengths) or not source_weights or
                    set(trajectory_source_labels) != set(source_weights) or
                    any(not isinstance(k, str) or not k for k in source_weights) or
                    any(not np.isfinite(v) or v <= 0 for v in source_weights.values())):
                raise ValueError("Expert sources require one known label per trajectory and positive finite weights")
            self.source_names = list(source_weights)
            self.trajectory_source_ids = torch.tensor([self.source_names.index(k) for k in trajectory_source_labels], device=device)
            self._source_weights = torch.tensor(list(source_weights.values()), device=device, dtype=torch.float32)
            self._raw_priorities = torch.ones_like(self.priorities)
            self.priorities = self._source_priorities(torch.ones_like(self.lengths, dtype=torch.bool))
        self.trajectory_morphology_delta_m = None
        if "morphology_delta_m" in self.storage:
            delta = self.storage["morphology_delta_m"]
            if self.payload_device is not None:
                # Only small sampling metadata is copied; observation payload stays on CPU.
                delta = delta.to(self.device)
            if delta.ndim != 2 or not bool(torch.isfinite(delta).all()):
                raise ValueError("Expert morphology must be finite [frames, dimensions]")
            self.trajectory_morphology_delta_m = delta[self.start_idx[:, 0]].clone()
            for start, length, shape in zip(self.start_idx[:, 0], self.lengths, self.trajectory_morphology_delta_m):
                frames = (start + torch.arange(int(length), device=self.device)) % self.capacity
                if not bool(torch.isclose(delta[frames], shape[None], atol=1e-7, rtol=0).all()):
                    raise ValueError("Expert morphology must stay constant within each complete trajectory")
        self._get_idxs = torch.compile(get_idxs, mode="reduce-overhead")

    def sample(self, batch_size: int = 1, seq_length: int | None = None, morphology_delta_m: torch.Tensor | None = None):
        seq_length = seq_length or self.seq_length
        if batch_size < seq_length:
            raise ValueError(
                f"The batch-size must be bigger than the sequence length, got batch_size={batch_size} and seq_length={seq_length}."
            )

        if batch_size % seq_length != 0:
            raise ValueError(
                f"The batch-size must be divisible by the sequence length, got batch_size={batch_size} and seq_length={seq_length}."
            )
        output, offset = {}, 0
        if len(self.output_key_tp1) > 0:
            output["next"] = {}
            offset = 1
        num_slices = batch_size // seq_length
        traj_idx = self.lengths >= (seq_length + offset)
        if morphology_delta_m is None:
            idxs = self._get_idxs(
                seq_length=seq_length,
                num_slices=num_slices,
                lengths=self.lengths[traj_idx],
                start_idx=self.start_idx[traj_idx],
                storage_length=self.capacity,
                priorities=self._source_priorities(traj_idx) if self.source_weights is not None else self.priorities[traj_idx],
            )
        else:
            if self.trajectory_morphology_delta_m is None:
                raise ValueError("Matched expert sampling requires explicit morphology_delta_m metadata")
            requested = torch.as_tensor(morphology_delta_m, device=self.device, dtype=self.trajectory_morphology_delta_m.dtype)
            if requested.shape != (num_slices, self.trajectory_morphology_delta_m.shape[1]) or not bool(torch.isfinite(requested).all()):
                raise ValueError("One finite morphology vector is required per requested expert sequence")
            shapes, assignment = torch.unique(requested, dim=0, return_inverse=True)
            idxs = torch.empty((batch_size, self.start_idx.shape[1]), device=self.device, dtype=torch.long)
            for shape_id, shape in enumerate(shapes):
                eligible = traj_idx & torch.isclose(self.trajectory_morphology_delta_m, shape[None], atol=1e-7, rtol=0).all(-1)
                if not bool(eligible.any()):
                    raise ValueError(f"No expert trajectory of morphology {shape.tolist()} is long enough for {seq_length} steps")
                slices = (assignment == shape_id).nonzero(as_tuple=False).flatten()
                priorities = self._source_priorities(eligible) if self.source_weights is not None else self.priorities[eligible]
                if float(priorities.sum()) <= 0:
                    priorities = torch.ones_like(priorities)
                # The matched path has a small bank of shapes. Avoid compiling a
                # new variable-sized graph for each reset batch / subset.
                selected = get_idxs(seq_length, len(slices), self.lengths[eligible],
                                    self.start_idx[eligible], self.capacity, priorities)
                rows = (slices[:, None] * seq_length + torch.arange(seq_length, device=self.device)).flatten()
                idxs[rows] = selected
        idxs = idxs.to(torch.long).unbind(-1)
        gather_idxs = tuple(i.cpu() for i in idxs) if self.payload_device is not None else idxs
        for k in self.output_key_t:
            output[k] = tree_map(lambda x: x[gather_idxs].to(self.device) if self.payload_device is not None else x[gather_idxs], self.storage[k])
        # increment the time index to get the next states
        idxs = ((idxs[0] + 1) % self.capacity, *idxs[1:])
        gather_idxs = tuple(i.cpu() for i in idxs) if self.payload_device is not None else idxs
        for k in self.output_key_tp1:
            output["next"][k] = tree_map(lambda x: x[gather_idxs].to(self.device) if self.payload_device is not None else x[gather_idxs], self.storage[k])
        return output

    def update_priorities(self, priorities: torch.Tensor, idxs: torch.Tensor) -> None:
        """update priorities of trajectories"""
        assert len(priorities) == len(self.priorities)
        if self.source_weights is not None:
            priorities = priorities.to(device=self.device, dtype=self.priorities.dtype)
            idxs = torch.as_tensor(idxs, device=self.device, dtype=torch.long)
            if (len(idxs) != len(self.priorities) or len(torch.unique(idxs)) != len(idxs) or
                    bool((idxs < 0).any()) or bool((idxs >= len(self.priorities)).any()) or
                    not bool(torch.isfinite(priorities).all()) or bool((priorities < 0).any())):
                raise ValueError("Invalid expert source priorities or trajectory indices")
            self._raw_priorities[idxs] = priorities
            self.priorities = self._source_priorities(torch.ones_like(self.lengths, dtype=torch.bool))
            return
        self.priorities[idxs] = priorities
        self.priorities = self.priorities / torch.sum(self.priorities)

    def _source_priorities(self, eligible):
        """Preserve bank mass; absent banks are excluded for conditional queries.

        A zero-priority bank falls back to uniform within that bank, matching
        MotionLib. Sampling remains stochastic, not an exact per-batch quota.
        """
        ids, raw = self.trajectory_source_ids[eligible], self._raw_priorities[eligible]
        result = torch.zeros_like(raw)
        for source_id, weight in enumerate(self._source_weights):
            mask = ids == source_id
            if not bool(mask.any()):
                continue
            values = raw[mask]
            if float(values.sum()) <= 0:
                values = torch.ones_like(values)
            result[mask] = weight * values / values.sum()
        if float(result.sum()) <= 0:
            raise ValueError("No eligible expert trajectory in any configured source")
        return result / result.sum()

    @property
    def capacity(self):
        return get_key(self.storage, self.end_key).shape[0]

    def __len__(self) -> int:
        return self.capacity

    def empty(self) -> bool:
        return len(self) == 0


@dataclasses.dataclass(kw_only=True)
class TrajectoryDictBufferMultiDim(DictBuffer):
    # DictBuffer that supports multiple dimensions + sampling consecutive subsequences. Can be used to store whole episodes
    n_dim: int = 1  # use n_dim 2 to support multi-dimensional data
    seq_length: int = 1
    output_key_t: List[str] = dataclasses.field(default_factory=lambda: ["observation"])
    output_key_tp1: List[str] = dataclasses.field(default_factory=lambda: ["observation"])
    end_key: Tuple[str] | str = "done"

    def __post_init__(self) -> None:
        self.storage = None
        self._idx = 0
        self._is_full = False
        self._recompute_start_stop = True
        self._get_idxs = torch.compile(get_idxs, mode="reduce-overhead", fullgraph=True)
        assert self.n_dim == 1 or self.n_dim == 2, "n_dim must be either 1 or 2 for TrajectoryDictBufferMultiDim"

    def _ndim(self) -> int:
        return self.n_dim

    def size(self):
        return len(self) * self.storage[self.end_key].shape[1]

    @torch.no_grad
    def extend(self, data: Dict) -> None:
        """Extend the buffer with new data.
        We use a dictionary representation for the storage."""
        super().extend(data)
        self._recompute_start_stop = True

    def sample(self, batch_size: int = 1, seq_length: int | None = None):
        seq_length = seq_length or self.seq_length
        if batch_size < seq_length:
            raise ValueError(
                f"The batch-size must be bigger than the sequence length, got batch_size={batch_size} and seq_length={seq_length}."
            )

        if batch_size % seq_length != 0:
            raise ValueError(
                f"The batch-size must be divisible by the sequence length, got batch_size={batch_size} and seq_length={seq_length}."
            )

        if self._recompute_start_stop:
            done = get_key(self.storage, self.end_key)
            if self.payload_device is not None:
                done = done.to(self.device)
            self.start_idx, self.stop_idx, self.lengths = find_start_stop_traj(
                done.squeeze()[: len(self)], at_capacity=self._is_full, cursor=self._idx - 1
            )
            self._recompute_start_stop = False

        output, offset = {}, 0
        if len(self.output_key_tp1) > 0:
            output["next"] = {}
            offset = 1
        num_slices = batch_size // seq_length
        traj_idx = self.lengths >= (seq_length + offset)
        if not traj_idx.any():
            raise ValueError(f"No trajectories with length >= {seq_length + offset} to sample {num_slices} slices.")
        idxs = self._get_idxs(
            seq_length=seq_length,
            num_slices=num_slices,
            lengths=self.lengths[traj_idx],
            start_idx=self.start_idx[traj_idx],
            storage_length=self.capacity,
            priorities=None,
        )
        idxs = idxs.to(torch.long).unbind(-1)
        gather_idxs = tuple(i.cpu() for i in idxs) if self.payload_device is not None else idxs
        for k in self.output_key_t:
            output[k] = tree_map(lambda x: x[gather_idxs].to(self.device) if self.payload_device is not None else x[gather_idxs], self.storage[k])
        # increment the time index to get the next states
        idxs = ((idxs[0] + 1) % self.capacity, *idxs[1:])
        gather_idxs = tuple(i.cpu() for i in idxs) if self.payload_device is not None else idxs
        for k in self.output_key_tp1:
            output["next"][k] = tree_map(lambda x: x[gather_idxs].to(self.device) if self.payload_device is not None else x[gather_idxs], self.storage[k])
        return output

    def get_full_buffer(self) -> Dict:
        """We assume to return transition based"""
        if self.payload_device is not None:
            raise NotImplementedError("CPU payload full-buffer relabeling is not qualified")

        if self._recompute_start_stop:
            done = get_key(self.storage, self.end_key)
            if self.payload_device is not None:
                done = done.to(self.device)
            self.start_idx, self.stop_idx, self.lengths = find_start_stop_traj(
                done.squeeze()[: len(self)], at_capacity=self._is_full, cursor=self._idx - 1
            )
            self._recompute_start_stop = False

        output, offset = {}, 0
        if len(self.output_key_tp1) > 0:
            output["next"] = {}
            offset = 1
        for start, length in zip(self.start_idx, self.lengths):
            idxs = (start[0] + torch.arange(length - offset, device=self.device)) % self.capacity
            for k in self.output_key_t:
                if k not in output:
                    output[k] = []
                output[k].append(tree_map(lambda x: x[idxs, start[1]], self.storage[k]))
            idxs = (start[0] + 1 + torch.arange(length - offset, device=self.device)) % self.capacity
            for k in self.output_key_tp1:
                if k not in output["next"]:
                    output["next"][k] = []
                output["next"][k].append(tree_map(lambda x: x[idxs, start[1]], self.storage[k]))
        for k in self.output_key_t:
            output[k] = tree_concat(output[k])
        for k in self.output_key_tp1:
            output["next"][k] = tree_concat(output["next"][k])
        return output


def get_idxs(seq_length, num_slices, lengths, start_idx, storage_length, priorities: torch.Tensor | None = None):
    if priorities is not None:
        traj_idx = torch.multinomial(priorities, num_slices, replacement=True)
    else:
        traj_idx = torch.randint(lengths.shape[0], (num_slices,), device=lengths.device)
    end_point = lengths[traj_idx] - seq_length - 1
    relative_starts = (torch.rand(num_slices, device=lengths.device) * end_point).floor().to(start_idx.dtype)
    starts = torch.cat(
        [
            (start_idx[traj_idx, 0] + relative_starts).unsqueeze(1),
            start_idx[traj_idx, 1:],
        ],
        1,
    )
    idxs = _tensor_slices_from_startend(seq_length, starts, storage_length=storage_length)
    return idxs


def find_start_stop_traj(end: torch.Tensor, at_capacity: bool = True, cursor=None | int):
    length = end.shape[0]
    if not at_capacity:
        end = torch.index_fill(
            end,
            index=torch.tensor(length - 1, device=end.device, dtype=torch.long),
            dim=0,
            value=1,
        )
    else:
        if cursor is not None:
            if not _is_int(cursor):
                raise ValueError("cursor must be an int")
            end = torch.index_fill(
                end,
                index=torch.tensor(cursor, device=end.device, dtype=torch.long),
                dim=0,
                value=1,
            )
        if not end.any(0).all():
            mask = ~end.any(0, True)
            mask = torch.cat([torch.zeros_like(end[:-1]), mask])
            end = torch.masked_fill(mask, end, 1)
    return _end_to_start_stop(length=length, end=end)


def _end_to_start_stop(end, length):
    # Using transpose ensures the start and stop are sorted the same way
    stop_idx = end.transpose(0, -1).nonzero()
    stop_idx[:, [0, -1]] = stop_idx[:, [-1, 0]].clone()
    # First build the start indices as the stop + 1, we'll shift it later
    start_idx = stop_idx.clone()
    start_idx[:, 0] += 1
    start_idx[:, 0] %= end.shape[0]
    # shift start: to do this, we check when the non-first dim indices are identical
    # and get a mask like [False, True, True, False, True, ...] where False means
    # that there's a switch from one dim to another (ie, a switch from one element of the batch
    # to another). We roll this one step along the time dimension and these two
    # masks provide us with the indices of the permutation matrix we need
    # to apply to start_idx.
    if start_idx.shape[0] > 1:
        start_idx_mask = (start_idx[1:, 1:] == start_idx[:-1, 1:]).all(-1)
        m1 = torch.cat([torch.zeros_like(start_idx_mask[:1]), start_idx_mask])
        m2 = torch.cat([start_idx_mask, torch.zeros_like(start_idx_mask[:1])])
        start_idx_replace = torch.empty_like(start_idx)
        start_idx_replace[m1] = start_idx[m2]
        start_idx_replace[~m1] = start_idx[~m2]
        start_idx = start_idx_replace
    else:
        # In this case we have only one start and stop has already been set
        pass
    lengths = stop_idx[:, 0] - start_idx[:, 0] + 1
    lengths[lengths <= 0] = lengths[lengths <= 0] + length
    return start_idx, stop_idx, lengths


def _tensor_slices_from_startend(seq_length, starts, storage_length):
    arange = torch.arange(seq_length, device=starts.device, dtype=starts.dtype)
    ndims = starts.shape[-1] - 1 if (starts.ndim - 1) else 0
    if ndims:
        arange_reshaped = torch.empty(list(arange.shape) + [ndims + 1], device=starts.device, dtype=starts.dtype)
        arange_reshaped[..., 0] = arange
        arange_reshaped[..., 1:] = 0
    else:
        arange_reshaped = arange.unsqueeze(-1)
    arange_expanded = arange_reshaped.expand([starts.shape[0]] + list(arange_reshaped.shape))
    n_missing_dims = arange_expanded.dim() - starts.dim()
    start_expanded = starts[(slice(None),) + (None,) * n_missing_dims].expand_as(arange_expanded)
    result = (start_expanded + arange_expanded).flatten(0, 1)
    result[:, 0] = result[:, 0] % storage_length
    return result


def key_exists(data, key):
    if isinstance(key, str):
        return key in data.keys()
    else:
        if len(key) == 1:
            return key[0] in data.keys()
        if key[0] not in data.keys():
            return False
        return key_exists(data[key[0]], key[1:])


def set_key(data, key, value):
    if isinstance(key, str):
        data[key] = value
    elif len(key) == 1:
        data[key[0]] = value
    else:
        set_key(data, key[1:], value)


def get_key(data, key):
    if isinstance(key, str):
        return data[key]
    elif len(key) == 1:
        return data[key[0]]
    else:
        return get_key(data, key[1:])
