"""Fixed metadata writers: explicit 4 MiB failure allowance, not a size prediction."""
from __future__ import annotations
import os
from pathlib import Path
import stat

MAX_BYTES = 4 * 1024**2
NAMES = {
    "work_yaml": "config.yaml", "work_json": "config.json",
    "warm_start": "warm_start_receipt.json", "progress": "training_progress.json",
    "asset_yaml": "scaler_ufo_robot.yaml", "asset_json": "morphology_metadata.json",
}


def checked_path(path, kind):
    path = Path(path)
    if kind not in NAMES or path.name != NAMES[kind]:
        raise ValueError("Unknown fixed metadata producer/path")
    if not path.is_absolute() or path.parent != path.parent.resolve(strict=True):
        raise ValueError("Metadata parent must be canonical and existing")
    if os.path.lexists(path):
        s = path.lstat()
        if not stat.S_ISREG(s.st_mode) or s.st_nlink != 1 or s.st_size > MAX_BYTES:
            raise ValueError("Metadata old file type/link/size outside allowance")
    return path


def encode_text(path, text, *, kind):
    checked_path(path, kind)
    if type(text) is not str:
        raise TypeError("Original serialized metadata text required")
    raw = text.encode("utf-8")
    if len(raw) > MAX_BYTES:
        raise ValueError("Complete UTF8 metadata exceeds 4 MiB operation allowance")
    return raw


def write_text(path, text, *, kind):
    raw = encode_text(path, text, kind=kind)  # before create or truncate
    path = checked_path(path, kind)
    old = path.lstat() if path.exists() else None
    flags = os.O_WRONLY | os.O_NOFOLLOW | (0 if old else os.O_CREAT | os.O_EXCL)
    fd = os.open(path, flags, 0o600)
    try:
        current = os.fstat(fd)
        if not stat.S_ISREG(current.st_mode) or current.st_nlink != 1:
            raise ValueError("Metadata FD type/link differs")
        if old is not None and (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns, current.st_ctime_ns) != (old.st_dev, old.st_ino, old.st_size, old.st_mtime_ns, old.st_ctime_ns):
            raise ValueError("Metadata identity changed before truncate")
        os.ftruncate(fd, 0)
        with os.fdopen(fd, "wb", closefd=False) as stream:
            if stream.write(raw) != len(raw):
                raise OSError("Short metadata write")
            stream.flush()
    finally:
        os.close(fd)
    # No success receipt/durability claim or aggregate schedule reset is created.
    return len(text)


def read_text(path, *, kind):
    path = checked_path(path, kind)
    before = path.lstat()
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as stream:
        fs = os.fstat(stream.fileno())
        if (fs.st_dev, fs.st_ino) != (before.st_dev, before.st_ino):
            raise ValueError("Metadata changed before read")
        raw = stream.read(MAX_BYTES + 1)
    after = path.lstat()
    fields = ("st_dev", "st_ino", "st_nlink", "st_size", "st_mtime_ns", "st_ctime_ns")
    if len(raw) > MAX_BYTES or len(raw) != before.st_size or any(getattr(before, k) != getattr(after, k) for k in fields):
        raise ValueError("Metadata changed or exceeded read allowance")
    return raw.decode("utf-8")
