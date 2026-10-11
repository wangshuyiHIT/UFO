"""Explicit model-state initialization; stdlib inspection never constructs a model.

Resume owns the target checkpoint and its clocks. Warm-start only reads a pinned
source generation and leaves every optimizer and rollout buffer freshly built.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import re
import struct

from .checkpoint_io import file_digest, validate_generation


def inspect_source(spec: dict) -> dict:
    required = {"generation", "manifest_sha256", "model_sha256"}
    receipt_keys = {"launch_receipt", "launch_receipt_sha256"}
    if set(spec) not in (required, required | receipt_keys):
        raise ValueError("Warm-start requires generation, manifest_sha256 and model_sha256")
    for key in ("manifest_sha256", "model_sha256", *(["launch_receipt_sha256"] if receipt_keys <= set(spec) else [])):
        if not isinstance(spec[key], str) or not re.fullmatch(r"[0-9a-f]{64}", spec[key]):
            raise ValueError("Warm-start requires exact SHA256 identities")
    requested = Path(spec["generation"])
    if not requested.is_absolute() or requested.is_symlink() or not re.fullmatch(r"generation-[0-9a-f]{32}", requested.name):
        raise ValueError("Warm-start requires an absolute immutable generation, not a pointer")
    source = requested.resolve(strict=True)
    if file_digest(source / "COMMITTED.json") != spec["manifest_sha256"]:
        raise ValueError("Warm-start COMMITTED SHA256 mismatch")
    if any(p.is_symlink() for p in source.rglob("*")):
        raise ValueError("Warm-start generation contains a symlink")
    manifest = validate_generation(source, full_hashes=True)
    expected = set(manifest["files"]) | {"COMMITTED.json"}
    expected_dirs = {str(p) for name in expected for p in Path(name).parents if str(p) != "."}
    actual = {str(p.relative_to(source)) for p in source.rglob("*") if p.is_file()}
    directories = {str(p.relative_to(source)) for p in source.rglob("*") if p.is_dir()}
    if actual != expected or directories != expected_dirs:
        raise ValueError("Warm-start generation has uncommitted files or directories")
    if "effective_train_config.json" not in manifest["files"]:
        raise ValueError("Warm-start requires a committed effective_train_config.json")
    if manifest["files"]["model/model.safetensors"]["sha256"] != spec["model_sha256"]:
        raise ValueError("Warm-start model SHA256 mismatch")
    with (source / "model/model.safetensors").open("rb") as stream:
        length = struct.unpack("<Q", stream.read(8))[0]
        if length > 16 * 1024 * 1024:
            raise ValueError("Unreasonable safetensors header length")
        tensors = json.loads(stream.read(length))
    effective_config = json.loads((source / "effective_train_config.json").read_text())
    asset_evidence = None
    if "launch_receipt" in spec:
        receipt_path = Path(spec["launch_receipt"])
        if not receipt_path.is_absolute() or file_digest(receipt_path) != spec["launch_receipt_sha256"]:
            raise ValueError("Warm-start launch receipt path/SHA256 mismatch")
        receipt = json.loads(receipt_path.read_text())
        command = receipt.get("command", [])
        if (receipt.get("schema") != "scaler_ufo_stage_launch_v1" or command.count("--work-dir") != 1 or
                command[command.index("--work-dir") + 1] != effective_config.get("work_dir") or
                not isinstance(receipt.get("asset_contract"), dict)):
            raise ValueError("Warm-start launch receipt does not describe the committed source run")
        asset_evidence = receipt["asset_contract"]
    return {"source": {**spec, "generation": str(source)},
            "agent_config": json.loads((source / "config.json").read_text()),
            "model_config": json.loads((source / "model/config.json").read_text()),
            "build_kwargs": json.loads((source / "init_kwargs.json").read_text()),
            "effective_config": effective_config,
            "source_asset_contract": asset_evidence,
            "source_train_status": json.loads((source / "train_status.json").read_text()),
            "tensor_header": {k: v for k, v in tensors.items() if k != "__metadata__"}}


def interface_contract(config: dict) -> dict:
    """Data/rank/optimizer settings may change; physical input semantics may not."""
    agent, env = config["agent"], config["env"]
    model = copy.deepcopy(agent["model"])
    for key in ("device", "inference_batch_size"):
        model.pop(key, None)
    robot = copy.deepcopy(env["robot_training"])
    robot.pop("config_path", None)
    robot["robot"].pop("xml_path", None)
    morphology = robot.get("morphology")
    if morphology:
        for key in ("allow_review_motions", "motion_purpose"):
            morphology.pop(key, None)
    # Noise/DR are run conditions. All other overrides (control, observation,
    # reward and reset conventions) remain part of the compatibility contract.
    overrides = [x for x in env.get("hydra_overrides", [])
                 if not x.lstrip("+").startswith(("domain_rand.", "obs.noise_scales."))]
    return {"agent": agent["name"], "model": model, "robot_training": robot,
            "observation": {k: env.get(k) for k in (
                "name", "relative_config_path", "include_last_action", "context_length",
                "include_history_actor", "include_history_noaction", "root_height_obs")},
            "semantic_overrides": overrides,
            "aux_rewards": agent.get("aux_rewards"), "aux_rewards_scaling": agent.get("aux_rewards_scaling"),
            "discount": agent["train"].get("discount")}


def validate_interface(source: dict, target_config: dict, *, build_kwargs: dict | None = None) -> None:
    if interface_contract(source["effective_config"]) != interface_contract(target_config):
        raise ValueError("Warm-start interface/physical semantics differ; implicit migration is forbidden")
    # The saved standalone agent configuration must agree with its committed
    # effective configuration, not merely have compatible tensor dimensions.
    if source["agent_config"] != source["effective_config"]["agent"]:
        raise ValueError("Warm-start source agent and effective configurations disagree")
    if source["model_config"] != source["agent_config"]["model"]:
        raise ValueError("Warm-start source model and agent configurations disagree")
    if build_kwargs is not None:
        # Round-trip tuples/lists without converting or reordering feature keys.
        normalized = json.loads(json.dumps(build_kwargs))
        if normalized != source["build_kwargs"]:
            raise ValueError("Warm-start observation/action spaces differ")


def assert_fresh_target(work_dir: Path) -> None:
    for name in ("checkpoint", "checkpoint_generations", "train_status.json", "training_progress.json", "warm_start_receipt.json"):
        path = work_dir / name
        if path.exists() or path.is_symlink():
            raise ValueError("Warm-start requires fresh training state; use normal resume for an existing run")


def initialize_model(agent, source: dict, target_config: dict) -> dict:
    """Build agent first; strict in-place load preserves fresh optimizer objects."""
    import torch
    import safetensors.torch
    from humanoidverse.agents.envs.utils.gym_spaces import space_to_json
    from humanoidverse.distributed import agent_optimizers

    validate_interface(source, target_config, build_kwargs={
        "obs_space": space_to_json(agent.obs_space), "action_dim": agent.action_dim})
    optimizers = agent_optimizers(agent)
    if not optimizers or any(optimizer.state for optimizer in optimizers):
        raise ValueError("Warm-start requires freshly constructed empty optimizers")
    state = agent._model.state_dict()
    header = source["tensor_header"]
    if set(state) != set(header) or any(list(v.shape) != header[k]["shape"] for k, v in state.items()):
        raise ValueError("Warm-start model tensor keys/shapes differ")
    path = Path(source["source"]["generation"]) / "model/model.safetensors"
    weights = safetensors.torch.load_file(path, device=str(next(agent._model.parameters()).device))
    for key, value in weights.items():
        if value.dtype != state[key].dtype or not torch.isfinite(value).all():
            raise ValueError(f"Warm-start tensor dtype/nonfinite mismatch: {key}")
    agent._model.load_state_dict(weights, strict=True)
    return {"schema": "ufo_model_state_warm_start_v1", **source["source"],
            "source_asset_contract": source["source_asset_contract"],
            "source_train_status": source["source_train_status"], "model_tensor_count": len(state),
            "loaded": "complete_model_including_targets_and_normalizers", "optimizers": "fresh",
            "replay_expert_context_and_run_counters": "fresh",
            "training_data_provenance": target_config.get("training_data_provenance")}
