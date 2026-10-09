from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .label_generator import LabelConfig, make_multitask_labels


class SteadMultitaskDataset(Dataset):
    """STEAD-style multitask dataset interface, smoke-tested on synthetic data first.

    The synthetic HDF5 layout is documented as /data/{trace_name} with arrays shaped
    [6000, 3] (samples, channels). The loader also accepts [3, 6000] arrays and
    converts all outputs to torch tensors shaped [3, 6000].
    """

    def __init__(
        self,
        metadata_csv: str | Path,
        waveform_hdf5: str | Path,
        split: str | None = None,
        normalize: str = "per_trace",
        label_config: LabelConfig | None = None,
    ) -> None:
        self.metadata_csv = Path(metadata_csv)
        self.waveform_hdf5 = Path(waveform_hdf5)
        self.normalize = normalize
        self.label_config = label_config or LabelConfig()
        self.metadata = pd.read_csv(self.metadata_csv)
        if "p_arrival_sample" not in self.metadata.columns and "trace_p_arrival_sample" in self.metadata.columns:
            self.metadata["p_arrival_sample"] = self.metadata["trace_p_arrival_sample"]
        if "s_arrival_sample" not in self.metadata.columns and "trace_s_arrival_sample" in self.metadata.columns:
            self.metadata["s_arrival_sample"] = self.metadata["trace_s_arrival_sample"]
        self._h5: h5py.File | None = None
        required = {"trace_name", "trace_category", "p_arrival_sample", "s_arrival_sample", "split"}
        missing = required.difference(self.metadata.columns)
        if missing:
            raise ValueError(f"Metadata is missing required columns: {sorted(missing)}")
        if split is not None:
            self.metadata = self.metadata[self.metadata["split"].astype(str) == split].reset_index(drop=True)
        if normalize not in {"per_trace", "none"}:
            raise ValueError("normalize must be 'per_trace' or 'none'")

    def __len__(self) -> int:
        return int(len(self.metadata))

    def _file(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.waveform_hdf5, "r")
        return self._h5

    def close(self) -> None:
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None

    def _read_waveform(self, trace_name: str) -> np.ndarray:
        h5 = self._file()
        if "data" in h5 and "$" in trace_name:
            bucket, selector = trace_name.split("$", 1)
            row_index = int(selector.split(",", 1)[0])
            if bucket not in h5["data"]:
                raise KeyError(f"Bucket {bucket!r} not found in {self.waveform_hdf5}")
            array = np.asarray(h5["data"][bucket][row_index], dtype=np.float32)
        elif "data" in h5 and trace_name in h5["data"]:
            array = np.asarray(h5["data"][trace_name], dtype=np.float32)
        elif trace_name in h5:
            array = np.asarray(h5[trace_name], dtype=np.float32)
        else:
            raise KeyError(f"Trace {trace_name!r} not found in {self.waveform_hdf5}")

        if array.shape == (self.label_config.window_length, 3):
            array = array.T
        elif array.shape == (3, self.label_config.window_length):
            pass
        else:
            raise ValueError(f"Expected [6000,3] or [3,6000], got {array.shape} for {trace_name}")

        if self.normalize == "per_trace":
            mean = array.mean(axis=1, keepdims=True)
            std = array.std(axis=1, keepdims=True)
            array = (array - mean) / np.maximum(std, 1e-6)
        return array.astype(np.float32)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.metadata.iloc[index]
        trace_name = str(row["trace_name"])
        waveform = self._read_waveform(trace_name)
        labels = make_multitask_labels(
            row["p_arrival_sample"],
            row["s_arrival_sample"],
            str(row["trace_category"]),
            config=self.label_config,
        )
        label_tensor = torch.stack(
            [
                torch.from_numpy(labels["detection"]),
                torch.from_numpy(labels["p"]),
                torch.from_numpy(labels["s"]),
            ],
            dim=0,
        )
        metadata = {key: row[key].item() if hasattr(row[key], "item") else row[key] for key in self.metadata.columns}
        return {
            "waveform": torch.from_numpy(waveform),
            "labels": label_tensor,
            "labels_dict": {key: torch.from_numpy(value) for key, value in labels.items()},
            "metadata": metadata,
        }

    def __del__(self) -> None:
        if hasattr(self, "_h5"):
            self.close()

