# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
"""Run scratchpad VLA checkpoints on ClevrSkills evaluation tasks."""

import gc
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from absl import app, flags
from peft import PeftModel
from tqdm import tqdm

try:
    from clevr_skills.utils.visualize_prompt import PromptVisualizer
except ModuleNotFoundError:
    PromptVisualizer = None

from eval_utils import (
    get_clevr_skills_task_wargs,
    get_clevrskills_env,
    get_preprocessed_image,
)
from paligemma_runner import PaliGemmaVLARunner
from utils import find_checkpoint

ACTION_DIM = 7
DEFAULT_VLA_VARIANT = "sp"
CLEVR_UNNORM_KEY = "clevr_skills_dataset"

logger = logging.getLogger(__name__)

flags.DEFINE_string(
    "finetuned_path",
    "checkpoints/place_next_restore/checkpoint",
    "Path to finetuned PaliGemma checkpoint directory.",
)
flags.DEFINE_string("vla_variant", DEFAULT_VLA_VARIANT, "PaliGemma VLA variant.")
flags.DEFINE_string("record_dir", "runs/eval", "Path to where to store eval trajectories.")
flags.DEFINE_integer("num_episodes", 10, "How many eval episodes to run.")
flags.DEFINE_integer("ep_length", 200, "How long to run each episode.")
flags.DEFINE_integer("start_seed", 500000, "First seed for the eval sequence.")
flags.DEFINE_integer("eval_seed", 0, "Random seed for evaluation.")
flags.DEFINE_string("task_description", "pickandplacenextto", "ClevrSkills task description.")
flags.DEFINE_integer("num_objects", 2, "Number of objects in the scene/task.")
flags.DEFINE_bool("language_winfo", True, "Add gripper information to language instruction.")
flags.DEFINE_bool("verbose", False, "Print decoded thoughts and actions.")
flags.DEFINE_bool("eval_latest", False, "Evaluate the latest checkpoint.")
flags.DEFINE_bool("eval_best", False, "Evaluate the best checkpoint.")
flags.DEFINE_bool("eval_fractional", False, "Allow evaluating fractional checkpoints.")
flags.DEFINE_integer("eval_epoch", -1, "Checkpoint epoch to evaluate.")

FLAGS = flags.FLAGS


def build_record_dir(
    record_dir: str,
    latest_ckpt_info: Dict[str, Any],
    eval_seed: int,
) -> str:
    """Build an output directory name that encodes checkpoint metadata."""
    if len(latest_ckpt_info) > 0:
        record_dir = record_dir.rstrip("/")
        info_str = "-".join(f"{key}={value}" for key, value in latest_ckpt_info.items())
        record_dir = f"{record_dir}_{info_str}"
    else:
        record_dir = f"{record_dir}_latest"
    if eval_seed != 0:
        record_dir = f"{record_dir}-seed={eval_seed}"
    return record_dir


def load_paligemma_model(finetuned_path: str) -> Tuple[PaliGemmaVLARunner, int]:
    """Load a PaliGemma runner and merge LoRA weights when present."""
    adapter_json = Path(finetuned_path) / "adapter_config.json"
    if adapter_json.is_file():
        with open(adapter_json, "r", encoding="utf-8") as f:
            adapter_config = json.load(f)
        model_path = adapter_config["base_model_name_or_path"]
    else:
        model_path = finetuned_path

    model = PaliGemmaVLARunner(
        model_path=model_path,
        finetuned_path=finetuned_path,
        vla_variant=FLAGS.vla_variant,
        do_sample=False,
    )

    if adapter_json.is_file():
        print("Adding LORA weights...")
        model.vla = PeftModel.from_pretrained(model.vla, finetuned_path)
        model.vla = model.vla.merge_and_unload()

    return model, 224


def make_language_addendum(obs: Dict[str, Any]) -> str:
    """Create optional language text that describes gripper state."""
    if not FLAGS.language_winfo:
        return ""

    num_suction_cups_ready = np.sum(obs["agent"]["vacuum_ready"])
    language_addendum = f"({num_suction_cups_ready} grippers ready"
    if obs["agent"]["vacuum_grasping"]:
        language_addendum += " and object grasped)"
    else:
        language_addendum += ")"
    return language_addendum


def format_scratchpad_for_video(
    scratchpad: Optional[Tuple[Optional[str], str, str]],
    include_plan: bool = False,
) -> Optional[str]:
    """Format scratchpad content for video overlays."""
    if scratchpad is None:
        return None

    plan, thought, _act_clause = scratchpad
    lines = []
    if include_plan and plan:
        lines.append(f"Plan: {' '.join(plan.split())}")
    if thought:
        lines.append(f"Thought: {' '.join(thought.split())}")
    if len(lines) == 0:
        return None
    return "\n".join(lines)


def predict_or_fallback(
    model: PaliGemmaVLARunner,
    image: np.ndarray,
    language_instruction: str,
    language_addendum: str,
    ep_start: bool,
) -> np.ndarray:
    """Predict an action, falling back to a small random action on model errors."""
    try:
        start_time = time.time()
        action = model.predict_action(
            image=image,
            instruction=language_instruction,
            language_addendum=language_addendum,
            unnorm_key=CLEVR_UNNORM_KEY,
            center_crop=True,
            verbose=FLAGS.verbose,
            ep_start=ep_start,
        )
        print("Time for inference step: ", time.time() - start_time)
        action[-1] = np.round(action[-1])
        return action
    except torch.cuda.OutOfMemoryError:
        print("| WARNING: ran out of memory, retrying batch")
        raise
    except (RuntimeError, ValueError, KeyError, IndexError) as exc:
        if hasattr(model, "last_scratchpad"):
            model.last_scratchpad = None
        print(exc)
        action = np.random.randn(ACTION_DIM) / 10
        if getattr(model, "prev_actions", None) is not None:
            action[3:] = model.prev_actions[3:]
        action[-1] = np.round(action[-1])
        print("ERROR IN ACTION PREDICTION: using small random action")
        return action


def save_model_traces(
    record_dir: str,
    task_name: str,
    seed: int,
    model: PaliGemmaVLARunner,
) -> None:
    """Save model thoughts and positions for one trajectory."""
    traj_dir = os.path.join(record_dir, task_name, f"traj_{seed}")
    if len(model.aux_thoughts) > 0:
        np.save(os.path.join(traj_dir, "thoughts.npy"), np.asarray(model.aux_thoughts))
    if len(model.aux_positions) > 0:
        np.save(os.path.join(traj_dir, "positions.npy"), np.asarray(model.aux_positions))


def load_existing_trajectory_metrics(
    traj_path: Union[str, Path],
) -> Optional[Tuple[float, Optional[float], Optional[float]]]:
    """Load metrics for an already-recorded trajectory, when available."""
    traj_path = Path(traj_path)
    success_path = traj_path / "success.npy"
    if not success_path.is_file():
        return None

    success_values = np.asarray(np.load(success_path, allow_pickle=False)).reshape(-1)
    success = float(success_values[-1]) if success_values.size > 0 else 0.0

    rewards_path = traj_path / "rewards.npy"
    if not rewards_path.is_file():
        return success, None, None

    rewards = np.asarray(np.load(rewards_path, allow_pickle=False), dtype=float).reshape(-1)
    episode_return = float(np.sum(rewards)) if rewards.size > 0 else 0.0
    reward_per_step = episode_return / max(int(rewards.size), 1)
    return success, episode_return, reward_per_step


def get_episode_prompt(env: Any) -> Any:
    """Return the first ClevrSkills prompt for the active episode."""
    eps_info = getattr(env.unwrapped, "_get_eps_info")()
    return eps_info["prompts"][0]


def flush_env(env: Any) -> None:
    """Flush trajectory/video buffers and clear local memory caches."""
    try:
        env.flush_trajectory()
        env.flush_video()
        torch.cuda.empty_cache()
        gc.collect()
    except (AttributeError, RuntimeError) as exc:
        logger.warning("Failed to flush environment outputs: %s", exc)


def save_eval_results(record_dir: str, results: Dict[str, Any]) -> None:
    """Save aggregate evaluation metrics as JSON."""
    os.makedirs(record_dir, exist_ok=True)
    results_path = os.path.join(record_dir, "results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, allow_nan=False)
    print("Saved results to: ", results_path)


def mean_or_none(values: Sequence[float]) -> Optional[float]:
    """Return the arithmetic mean or ``None`` for an empty sequence."""
    if len(values) == 0:
        return None
    return float(np.mean(values))


def main(_: Any) -> None:
    """Evaluate a selected scratchpad VLA checkpoint."""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if PromptVisualizer is None:
        raise ImportError("ClevrSkills must be installed to visualize prompts.")

    np.random.seed(FLAGS.eval_seed)
    torch.manual_seed(FLAGS.eval_seed)
    random.seed(FLAGS.eval_seed)

    if FLAGS.vla_variant != DEFAULT_VLA_VARIANT:
        raise NotImplementedError(f"Only sp is supported: {FLAGS.vla_variant}")

    finetuned_path, latest_ckpt_info = find_checkpoint(FLAGS.finetuned_path, FLAGS)
    print(latest_ckpt_info)
    print("Fine-tuned model path: ", finetuned_path)

    model, image_size = load_paligemma_model(finetuned_path)
    record_dir = build_record_dir(FLAGS.record_dir, latest_ckpt_info, FLAGS.eval_seed)
    print("Record dir: ", record_dir)

    task_name, task_args = get_clevr_skills_task_wargs(FLAGS.task_description)
    env = get_clevrskills_env(task_name, task_args, output_dir=record_dir)

    ep_success = []
    ep_rewards = []
    ep_rewards_per_step = []
    num_traj = 0
    val_seeds = [FLAGS.start_seed + i for i in range(FLAGS.num_episodes)]

    for seed in val_seeds:
        traj_path = os.path.join(record_dir, task_name, f"traj_{seed}")
        existing_metrics = load_existing_trajectory_metrics(traj_path)
        if existing_metrics is not None:
            success, episode_reward, reward_per_step = existing_metrics
            ep_success.append(success)
            if episode_reward is not None:
                ep_rewards.append(episode_reward)
            if reward_per_step is not None:
                ep_rewards_per_step.append(reward_per_step)
            num_traj += 1
            continue

        print("Traj ", seed)
        obs, info = env.reset(seed=seed, options={"record_dir": traj_path, "reconfigure": True})
        prompt = get_episode_prompt(env)
        language_instruction = PromptVisualizer().visualize_prompt(
            prompt, "", mode="natural_language", compose_image=False
        )[0]
        print("Current language instruction is ", language_instruction)

        episode_return = 0.0
        steps_taken = 0
        model.reset_aux()

        for step in tqdm(range(FLAGS.ep_length)):
            try:
                image = obs["image"]["base_camera"]["rgb"]
                image = get_preprocessed_image(image, resize_size=image_size)
                language_addendum = make_language_addendum(obs)
                action = predict_or_fallback(
                    model,
                    image,
                    language_instruction,
                    language_addendum,
                    ep_start=step == 0,
                )
                if FLAGS.verbose:
                    print("ACTION", action)
                scratchpad_text = format_scratchpad_for_video(
                    getattr(model, "last_scratchpad", None), include_plan=step == 0
                )
                extra_info = {}
                if scratchpad_text is not None:
                    extra_info["scratchpad"] = scratchpad_text
                obs, reward, done, trunc, info = env.step(action, extra_info)
            except KeyboardInterrupt:
                break

            steps_taken = step + 1
            episode_return += reward
            if done or trunc:
                break

        if steps_taken == 1 and info["success"]:
            print("Episode invalid!")
        else:
            ep_success.append(1.0 if info["success"] else 0.0)
            num_traj += 1
            ep_rewards.append(float(episode_return))
            divisor = max(steps_taken, 1)
            ep_rewards_per_step.append(float(episode_return / divisor))
            print(f"Episode return: {episode_return}")
            save_model_traces(record_dir, task_name, seed, model)

        flush_env(env)

    results = {
        "num_admissible_trajectories": num_traj,
        "success_rate": mean_or_none(ep_success),
        "avg_reward": mean_or_none(ep_rewards),
        "avg_reward_per_step": mean_or_none(ep_rewards_per_step),
        "successes": ep_success,
        "rewards": ep_rewards,
        "rewards_per_step": ep_rewards_per_step,
        "seeds": val_seeds,
        "record_dir": record_dir,
        "finetuned_path": finetuned_path,
        "checkpoint_info": latest_ckpt_info,
    }

    print("Number of admissible trajectories: ", results["num_admissible_trajectories"])
    print("Success rate: ", results["success_rate"])
    print("Avg reward: ", results["avg_reward"])
    print("Avg reward per step: ", results["avg_reward_per_step"])
    save_eval_results(record_dir, results)

    env.close()


if __name__ == "__main__":
    app.run(main)
