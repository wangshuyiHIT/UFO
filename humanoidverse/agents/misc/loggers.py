# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.


import dataclasses
import numpy as np
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import mediapy

import pandas as pd
from . import csv_output

def logfile_to_video_directory(logfile: str | Path):
    log_dir = Path(logfile).parent
    video_dir = log_dir / "videos"
    video_dir.mkdir(exist_ok=True)
    return video_dir

@dataclasses.dataclass
class CSVLogger:
    filename: Union[str, Path]
    fields: Optional[List[str]] = None

    def log(self, log_data: Dict[str, Any]) -> None:
        write_header = self.fields is None and not Path(self.filename).exists()
        if self.fields is None:
            self.fields = sorted(list(log_data.keys()))

        data = {field: log_data.get(field, "") for field in self.fields}  # Ensure all fields are present
        islist = [isinstance(v, Iterable) and not isinstance(v, str) for k, v in data.items()]
        if all(islist):
            df = pd.DataFrame(data)
        elif not any(islist):
            df = pd.DataFrame([data])
        else:
            raise RuntimeError("Fields should all be a numbers, a string or iterable objects. We don't support mixed types.")
        csv_output.append(self.filename, self.fields, df, write_header=write_header)

    def log_video(self, filename: str, frames: list[np.ndarray], fps: int) -> None:
        # Implement video logging logic here
        output_path = logfile_to_video_directory(self.filename) / filename
        # breakpoint()  # use PYTHONBREAKPOINT=0 to disable, or install ipdb for a nicer debugger
        mediapy.write_video(output_path, frames, fps=fps)


@dataclasses.dataclass
class JSONLogger:
    filename: Union[str, Path]
    fields: Optional[List[str]] = None

    def log(self, log_data: Dict[str, Any]) -> None:
        if self.fields is None:
            self.fields = sorted(list(log_data.keys()))
            if not Path(self.filename).exists():
                with open(self.filename, "w+") as f:
                    json.dump({k: [] for k in self.fields}, f)

        # not the most efficient way of logging since we cannot append
        with open(self.filename, "r+") as f:
            logz = json.load(f)
        with open(self.filename, "w+") as f:
            for field in self.fields:
                logz[field].append(log_data.get(field, ""))
            json.dump(logz, f)
