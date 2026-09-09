"""Checkpoint path helpers for scratchpad VLA evaluation."""

# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import logging
from ast import literal_eval
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple, Union

logger = logging.getLogger(__name__)

__all__ = [
    "CheckpointNotFoundError",
    "extract_ckpt_info",
    "find_checkpoint",
    "simplest_type",
]


class CheckpointNotFoundError(FileNotFoundError):
    """Raised when a requested checkpoint cannot be resolved."""


def simplest_type(s: str) -> Any:
    """Parse a string into the simplest Python literal type when possible."""
    try:
        return literal_eval(s)
    except (SyntaxError, ValueError):
        return s


def extract_ckpt_info(ckpt_path: Union[str, Path]) -> Dict[str, Any]:
    """Extract key-value metadata from a checkpoint directory name."""
    info = Path(ckpt_path).name
    infos = info.split("_")
    info_dict: Dict[str, Any] = {}
    for item in infos:
        k, v = item.split("=", 1)
        v = simplest_type(v)
        info_dict[k] = v
    return info_dict


def _checkpoint_root(finetuned_path: Path) -> Path:
    """Return the checkpoints directory associated with a requested path."""
    parts = finetuned_path.parts
    if "checkpoints" in parts:
        index = parts.index("checkpoints")
        return Path(*parts[: index + 1])
    return finetuned_path / "checkpoints"


def _non_empty_checkpoint(
    checkpoint_paths: Iterable[Path],
    allow_fractional: bool,
) -> Tuple[Path, Dict[str, Any]]:
    """Return the first non-empty checkpoint accepted by the caller."""
    for checkpoint_path in checkpoint_paths:
        latest_ckpt_info = extract_ckpt_info(checkpoint_path)
        if not isinstance(latest_ckpt_info["epoch"], int):
            logger.info("Fractional checkpoint %s", checkpoint_path)
            if not allow_fractional:
                continue

            logger.info("Using fractional checkpoint %s", checkpoint_path)
            latest_ckpt_info["epoch"] = int(latest_ckpt_info["epoch"])
            latest_ckpt_info["fractional"] = True
            latest_ckpt_info.pop("val-loss", None)
            latest_ckpt_info.pop("grad-step", None)

        if any(checkpoint_path.iterdir()):
            return checkpoint_path, latest_ckpt_info
        logger.info("Empty folder: %s", checkpoint_path)

    raise CheckpointNotFoundError("No non-empty checkpoint matched the request")


def find_checkpoint(
    finetuned_path: Union[str, Path],
    flags_obj: Any,
) -> Tuple[str, Dict[str, Any]]:
    """Resolve the checkpoint path and metadata requested by evaluation flags."""
    direct_checkpoint_markers = [
        "config.json",
        "dataset_statistics.json",
        "model.safetensors.index.json",
    ]
    finetuned_path = Path(finetuned_path)
    if finetuned_path.is_dir() and any(
        (finetuned_path / marker).is_file() for marker in direct_checkpoint_markers
    ):
        return str(finetuned_path), {}

    ckpt_dir = _checkpoint_root(finetuned_path)
    latest_ckpt_info: Dict[str, Any]
    if ckpt_dir.is_dir():
        all_ckpt_dirs = [path for path in ckpt_dir.iterdir() if path.is_dir()]

        if flags_obj.eval_latest:
            step2dir = {extract_ckpt_info(path)["grad-step"]: path for path in all_ckpt_dirs}
            ordered_paths = [step2dir[step] for step in sorted(step2dir, reverse=True)]
            finetuned_path, latest_ckpt_info = _non_empty_checkpoint(
                ordered_paths,
                allow_fractional=flags_obj.eval_fractional,
            )
        elif flags_obj.eval_best:
            loss2dir = {extract_ckpt_info(path)["val-loss"]: path for path in all_ckpt_dirs}
            ordered_paths = [loss2dir[loss] for loss in sorted(loss2dir)]
            finetuned_path, latest_ckpt_info = _non_empty_checkpoint(
                ordered_paths,
                allow_fractional=False,
            )
        elif flags_obj.eval_epoch > 0:
            epoch2dir = {extract_ckpt_info(path)["epoch"]: path for path in all_ckpt_dirs}
            try:
                finetuned_path = epoch2dir[flags_obj.eval_epoch]
            except KeyError as exc:
                raise CheckpointNotFoundError(
                    f"Checkpoint for epoch={flags_obj.eval_epoch} " f"not available at {ckpt_dir}"
                ) from exc
            latest_ckpt_info = extract_ckpt_info(finetuned_path)
        else:
            try:
                latest_ckpt_info = extract_ckpt_info(finetuned_path)
            except ValueError as exc:
                if finetuned_path.is_file():
                    latest_ckpt_info = {}
                else:
                    raise CheckpointNotFoundError(
                        f"Checkpoint not available at {ckpt_dir}"
                    ) from exc
    else:
        raise CheckpointNotFoundError(f"Checkpoint directory not available at {ckpt_dir}")
    return str(finetuned_path), latest_ckpt_info
