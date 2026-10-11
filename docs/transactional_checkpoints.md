# Transactional checkpoints

Enable `--transactional-checkpoints --no-checkpoint-buffer` for a new run whose
disk cannot hold distributed replay snapshots. This is opt-in: existing G1 runs
keep their original checkpoint behavior. A real legacy `checkpoint` directory
cannot be overwritten in transactional mode; use a new work directory. Existing
legacy directories remain readable. Older `checkpoint/CURRENT` generations are
readable but not writable by this implementation.

Each save writes an independent `checkpoint_generations/generation-<id>`.
The agent, optimizer, training counters, effective training config and optional
rank-local replay shards must all finish before `checkpoint` is atomically
replaced with a relative symlink to that generation. Standard inference paths
through `checkpoint/model` therefore continue to work. The manifest contains
file sizes and SHA256 hashes. Resume checks all sizes and small-file hashes;
`validate_generation(path, full_hashes=True)` also verifies large files.

Only the current pointer is a resume candidate. A newer orphan from an interrupted
save is never selected automatically. A corrupted current generation produces an
error rather than silently selecting older weights. After publication, the current
and its actual predecessor are retained. An interrupted writer can leave an
unreferenced directory; a subsequent successful save removes it. Disk preflight
does not delete artifacts to make space for a save.

Before writing, free space must exceed 1.2 times the larger of the serialized
tensor estimate and the last generation's measured size, plus
`--checkpoint-reserve-gib` (default 4 GiB). This is additional free space for one
in-flight generation; the two retained generations already occupy disk. A cloud
deployment should also check total capacity for two retained generations plus
one in-flight generation before starting. For the measured G1 model+optimizer,
three generations with 20% headroom require about 23.2 GiB. Scaler sizes must be
measured during the initial short run.

Distributed saves use a separate CPU Gloo control group, with
`--checkpoint-timeout-seconds` (default 300). Every live rank reports each phase's
success or exception. Rank 0 alone writes shared model/status files; replay shards
are separate for every rank. Publication requires all ranks' write success. Local
SHA256 summaries require exact model equality at a checkpoint; a mismatch stops
the save while keeping the old generation. The timeout bounds a peer waiting for a
stalled writer; it does not interrupt an operating-system write in the writer
itself. The cloud supervisor's hard deadline remains necessary.

When a transactional run resumes without replay, it collects `num_seed_steps`
new local environment steps before any optimizer update. It uses the restored
policy during this warmup and retains saved total and optimizer counters. If any
rank lacks replay, all ranks warm up together. Existing nonempty replay avoids
this extra warmup. A checkpoint without replay is a warm continuation, not a
bit-for-bit continuation: simulator state, replay, RNG state and transient rollout
context are not restored.

Validation: `./.venv/bin/python -m unittest discover -s tests -p
test_transactional_checkpoints.py -v` exercises retention, restore selection,
partial agent/status/publication failures, disk preflight, legacy protection,
corruption, writer locking, real two-rank failure propagation, bounded Gloo
timeout, CLI/config integration, saved counters and replay warmup. GPU and cloud
validation must be recorded separately.
