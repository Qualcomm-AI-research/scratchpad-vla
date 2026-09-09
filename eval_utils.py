# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
"""Utilities for ClevrSkills evaluation, image preprocessing, and plots."""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image

try:
    import clevr_skills  # noqa: F401
    import mani_skill2  # noqa: F401
    from clevr_skills.utils.record_env import RecordEnv
    from clevr_skills.utils.record_env_pretty import RecordPrettyEnv
    from clevr_skills.utils.visualize_prompt import PromptVisualizer
except ModuleNotFoundError:
    RecordEnv = None
    RecordPrettyEnv = None
    PromptVisualizer = None

ACTION_DIM = 7
IMAGE_TOKEN_COUNT = 256
LINEBREAK_ROWS = 8
DEFAULT_CROP_SCALE = 0.9
DEFAULT_CROP_BATCH_SIZE = 1


def get_clevrskills_env(
    task: str,
    task_args: Dict[str, Any],
    output_dir: Union[str, Path] = "outs",
    info_on_video: bool = False,
    pretty: bool = True,
) -> Any:
    """Create a ClevrSkills environment wrapped for trajectory recording."""
    if RecordEnv is None or RecordPrettyEnv is None or PromptVisualizer is None:
        raise ImportError(
            "ClevrSkills and ManiSkill2 must be installed to create an " "evaluation environment."
        )

    env = gym.make(
        "ClevrSkills-v0",
        obs_mode="rgbd",
        reward_mode="dense",
        control_mode="pd_ee_delta_pose",
        robot="xarm6_vacuum",
        task=task,
        strip_eval=False,
        shader_dir="ibl",
        render_config={},
        enable_shadow=True,
        task_args=task_args,
        render_mode="rgb_array",
    )

    record_env_class = RecordPrettyEnv if pretty else RecordEnv

    env = record_env_class(
        env,
        str(Path(output_dir) / task),
        save_trajectory=True,
        trajectory_name=f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}",
        save_video=True,
        save_on_reset=False,
        info_on_video=info_on_video,
        prompt_visualization_mode=PromptVisualizer.NATURAL_LANGUAGE,
        action_label_visualization_mode="vla_thoughts",
    )

    return env


def get_clevr_skills_task_wargs(
    task_description: str,
) -> Tuple[str, Dict[str, Any]]:
    """Map a short task description to ClevrSkills task name and args."""
    task_description = task_description.lower()
    if task_description == "touchpick":
        task_name = "TouchPick"
        task_args = {"split": "cs_mem"}
    elif task_description == "swap":
        task_name = "Swap"
        task_args = {"split": "cs_mem"}
    elif task_description == "stack_topple":
        task_name = "SingleStack"
        task_args = {"num_actors": 4, "topple": True, "split": "cs_mem"}
    elif task_description == "place_next_restore":
        task_name = "PlaceNextTo"
        task_args = {
            "num_actors": 3,
            "spawn_at_gripper": False,
            "restore": True,
            "split": "cs_mem",
        }
    elif task_description == "rotate_restore":
        task_name = "Rotate"
        task_args = {"num_actors": 3, "restore": True, "split": "cs_mem"}
    else:
        raise NotImplementedError
    return task_name, task_args


def normalize_gripper_action(
    action: np.ndarray,
    binarize: bool = True,
) -> np.ndarray:
    """Normalize the gripper action dimension from ``[0, 1]`` to ``[-1, 1]``."""
    # Just normalize the last action to [-1,+1].
    orig_low, orig_high = 0.0, 1.0
    action[..., -1] = 2 * (action[..., -1] - orig_low) / (orig_high - orig_low) - 1

    if binarize:
        # Binarize to -1 or +1.
        action[..., -1] = np.sign(action[..., -1])

    return action


def crop_and_resize(
    image: tf.Tensor,
    crop_scale: float,
    batch_size: int,
) -> tf.Tensor:
    """Center-crop an image and resize it back to model input size.

    Uses the same logic as the ``dlimp`` RLDS datasets wrapper to avoid
    distribution shift at test time.

    Args:
        image: Tensor shaped ``(batch_size, H, W, C)`` or ``(H, W, C)`` with
            dtype ``tf.float32`` and values between ``[0, 1]``.
        crop_scale: The area of the center crop with respect to the original image.
        batch_size: Batch size.
    """
    # Convert from 3D Tensor (H, W, C) to 4D Tensor (batch_size, H, W, C)
    assert image.shape.ndims in (3, 4)
    expanded_dims = False
    if image.shape.ndims == 3:
        image = tf.expand_dims(image, axis=0)
        expanded_dims = True

    # Get height and width of crop
    new_heights = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))
    new_widths = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))

    # Get bounding box representing crop
    height_offsets = (1 - new_heights) / 2
    width_offsets = (1 - new_widths) / 2
    bounding_boxes = tf.stack(
        [
            height_offsets,
            width_offsets,
            height_offsets + new_heights,
            width_offsets + new_widths,
        ],
        axis=1,
    )

    # Crop and then resize back up
    image = tf.image.crop_and_resize(image, bounding_boxes, tf.range(batch_size), (224, 224))

    # Convert back to 3D Tensor (H, W, C)
    if expanded_dims:
        image = image[0]

    return image


def resize_image(img: np.ndarray, resize_size: Tuple[int, int]) -> np.ndarray:
    """Resize one image using the training-time image resizing scheme."""
    assert isinstance(resize_size, tuple)
    # Resize to image size expected by model
    img = tf.image.encode_jpeg(img)  # Encode as JPEG, as done in RLDS dataset builder
    img = tf.io.decode_image(
        img, expand_animations=False, dtype=tf.uint8
    )  # Immediately decode back
    img = tf.image.resize(img, resize_size, method="lanczos3", antialias=True)
    img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)
    img = img.numpy()
    return img


def get_preprocessed_image(
    image: np.ndarray,
    resize_size: Union[int, Tuple[int, int]],
) -> np.ndarray:
    """Extracts image from observations and preprocesses it."""
    assert isinstance(resize_size, (int, tuple))
    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)
    image = resize_image(image, resize_size)
    return image


def plot_attention_heatmap(
    attn: Sequence[Any],
    input_ids: Any,
    tokens: Optional[list[Any]] = None,
    layer: int = 31,
    head: int = 0,
) -> None:
    """Plot full image/text/action attention for one layer and head."""
    if tokens is None:
        tokens = ["img" for _ in range(IMAGE_TOKEN_COUNT)]
        tokens.extend(input_ids[0].tolist())
        tokens.extend([f"act{i}" for i in range(ACTION_DIM - 1)])

    rows = [attn[0][layer][0, head, -1, :].cpu().detach().float()[None]]
    rows.extend(
        attn[i][layer][0, head, 0, :].cpu().detach().float()[None] for i in range(1, len(attn))
    )

    attention_map = np.zeros((ACTION_DIM, rows[-1].shape[1]))
    for i, row in enumerate(rows):
        attention_map[i, : row.shape[1]] = row

    _figure, ax = plt.subplots(figsize=(25, 3))

    heatmap = ax.imshow(attention_map, aspect="auto", cmap="viridis")

    ax.set_yticks(np.arange(ACTION_DIM))
    ax.set_yticklabels([f"act_{i}" for i in range(ACTION_DIM)], fontsize=8)

    ax.set_xticks(np.arange(len(tokens)))
    ax.set_xticklabels(tokens, rotation=60, fontsize=5)

    # Minor ticks
    ax.set_xticks(np.arange(-0.5, len(tokens), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, ACTION_DIM, 1), minor=True)

    # Gridlines based on minor ticks
    ax.grid(which="minor", color="w", linestyle="-", linewidth=0.5)

    # Remove minor ticks
    ax.tick_params(which="minor", bottom=False, left=False)

    ax.set_aspect("auto")
    plt.colorbar(heatmap, ax=ax, orientation="vertical")

    plt.tight_layout()
    plt.savefig(f"attention_layer{layer}_head{head}.png", dpi=600)
    plt.close()


def plot_attention_heatmap_reduced(
    attn: Sequence[Any],
    input_ids: Any,
    tokens: Optional[list[Any]] = None,
    layer: int = 31,
    head: int = 0,
) -> None:
    """Plot text/action attention with image-token positions omitted."""
    if tokens is None:
        tokens = input_ids[0].tolist()
        tokens.extend([f"act{i}" for i in range(ACTION_DIM - 1)])

    rows = [attn[0][layer][0, head, -1, IMAGE_TOKEN_COUNT:].cpu().detach().float()[None]]
    rows.extend(
        attn[i][layer][0, head, 0, IMAGE_TOKEN_COUNT:].cpu().detach().float()[None]
        for i in range(1, len(attn))
    )

    attention_map = np.zeros((ACTION_DIM, rows[-1].shape[1]))
    for i, row in enumerate(rows):
        attention_map[i, : row.shape[1]] = row

    _figure, ax = plt.subplots(figsize=(20, 4))

    heatmap = ax.imshow(attention_map, aspect="auto", cmap="viridis")

    ax.set_yticks(np.arange(ACTION_DIM))
    ax.set_yticklabels([f"act_{i}" for i in range(ACTION_DIM)], fontsize=8)

    ax.set_xticks(np.arange(len(tokens)))
    ax.set_xticklabels(tokens, rotation=60, fontsize=8)

    # Minor ticks
    ax.set_xticks(np.arange(-0.5, len(tokens), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, ACTION_DIM, 1), minor=True)

    # Gridlines based on minor ticks
    ax.grid(which="minor", color="w", linestyle="-", linewidth=2)

    # Remove minor ticks
    ax.tick_params(which="minor", bottom=False, left=False)

    ax.set_aspect("auto")
    plt.colorbar(heatmap, ax=ax, orientation="vertical")

    plt.tight_layout()
    plt.savefig(f"attention_layer{layer}_head{head}_prompt_only.png", dpi=600)
    plt.close()


def plot_attention_heatmap_linebreak(
    attn: Sequence[Any],
    input_ids: Any,
    tokens: Optional[list[Any]] = None,
    layer: int = 31,
    head: int = 0,
) -> None:
    """Plot attention for final prompt/action rows near a line break."""
    if tokens is None:
        tokens = ["img" for _ in range(IMAGE_TOKEN_COUNT)]
        tokens.extend(input_ids[0].tolist())
        tokens.extend([f"act{i}" for i in range(ACTION_DIM - 1)])

    rows = [
        attn[0][layer][0, head, -LINEBREAK_ROWS + i, :].cpu().detach().float()[None]
        for i in range(LINEBREAK_ROWS)
    ]

    attention_map = np.zeros((len(rows), rows[-1].shape[1]))
    for i, row in enumerate(rows):
        attention_map[i, : row.shape[1]] = row

    _figure, ax = plt.subplots(figsize=(25, 3))

    heatmap = ax.imshow(attention_map, aspect="auto", cmap="viridis")

    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(input_ids[0].tolist()[-LINEBREAK_ROWS:], fontsize=8)

    ax.set_xticks(np.arange(len(tokens)))
    ax.set_xticklabels(tokens, rotation=60, fontsize=5)

    # Minor ticks
    ax.set_xticks(np.arange(-0.5, len(tokens), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, ACTION_DIM, 1), minor=True)

    # Gridlines based on minor ticks
    ax.grid(which="minor", color="w", linestyle="-", linewidth=0.5)

    # Remove minor ticks
    ax.tick_params(which="minor", bottom=False, left=False)

    ax.set_aspect("auto")
    plt.colorbar(heatmap, ax=ax, orientation="vertical")

    plt.tight_layout()
    plt.savefig(f"attention_layer{layer}_head{head}_linebreak.png", dpi=600)
    plt.close()


def crop_and_resize_fn(
    image: Image.Image,
    crop_scale: float = DEFAULT_CROP_SCALE,
    batch_size: int = DEFAULT_CROP_BATCH_SIZE,
) -> Image.Image:
    """Center-crop a PIL image and return an RGB PIL image."""
    # Multiply original dimensions by sqrt(crop_scale), not crop_scale.

    # Convert to TF Tensor and record original data type (should be tf.uint8)
    image = tf.convert_to_tensor(np.array(image))
    orig_dtype = image.dtype

    # Convert to data type tf.float32 and values between [0,1]
    image = tf.image.convert_image_dtype(image, tf.float32)

    # Crop and then resize back to original size
    image = crop_and_resize(image, crop_scale, batch_size)

    # Convert back to original data type
    image = tf.clip_by_value(image, 0, 1)
    image = tf.image.convert_image_dtype(image, orig_dtype, saturate=True)

    # Convert back to PIL Image
    image = Image.fromarray(image.numpy())
    return image.convert("RGB")
