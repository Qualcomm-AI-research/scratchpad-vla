# Notes-to-Self: Scratchpad Augmented VLAs for Memory Dependent Manipulation Tasks

## Overview

Code for the paper [Notes-to-Self: Scratchpad Augmented VLAs for Memory Dependent Manipulation Tasks](https://arxiv.org/abs/2602.21013).

## Abstract
Many dexterous manipulation tasks are non-markovian in nature, yet little attention has been paid to this fact in the recent upsurge of the vision-language-action (VLA) paradigm. Although they are successful in bringing internet-scale semantic understanding to robotics, existing VLAs are primarily "stateless" and struggle with memory-dependent long horizon tasks. In this work, we explore a way to impart both spatial and temporal memory to a VLA by incorporating a language scratchpad. The scratchpad makes it possible to memorize task-specific information, such as object positions, and it allows the model to keep track of a plan and progress towards subgoals within that plan. We evaluate this approach on a split of memory-dependent tasks from the ClevrSkills environment, on MemoryBench, as well as on a challenging real-world pick-and-place task. We show that incorporating a language scratchpad significantly improves generalization on these tasks for both non-recurrent and recurrent models.

## Getting Started
Build the Docker image from the repository root:

```bash
docker build \
  --build-arg USER_UID="$(id -u)" \
  --build-arg USER_GID="$(id -g)" \
  -f docker/Dockerfile \
  --tag scratchpad-eval:latest .
```

The UID and GID build arguments make the image run as the account that built it,
which keeps bind-mounted files writable without creating an image user or running
the container as root. They default to `1000` when omitted.

Start an interactive container with the repository and Hugging Face cache mounted from your host:

```bash
mkdir -p "$HOME/.cache/huggingface"

docker run --rm -it \
  --gpus=all \
  --shm-size=32g \
  -v "$PWD":/workspace/scratchpad-vla \
  -v "$HOME/.cache/huggingface":/workspace/hf_cache \
  -w /workspace/scratchpad-vla \
  scratchpad-eval:latest \
  /bin/bash
```

This command starts the shell as the configured unprivileged UID/GID and mounts
the Hugging Face cache.

Inside the container, download the ClevrSkills resources before the first rollout:

```bash
cd /ClevrSkills
PYTHONPATH=/ClevrSkills bash scripts/download_resources.sh
cd /workspace/scratchpad-vla
```

Evaluation dependency defaults are centralized in [scripts/setup.sh](scripts/setup.sh).
The Docker image sets these paths automatically; for a non-Docker setup, update
`MANISKILL2_ROOT`, `CLEVRSKILLS_ROOT`, and `HF_HOME` there or export them before
running a task script.

### Download checkpoints

The evaluation scripts expect task checkpoints under `checkpoints/`. To download and unpack all released task checkpoints into that directory, run:

```bash
bash scripts/download_ckpts.sh
```

The task scripts default to local task checkpoints under `checkpoints/`. Override `FINETUNED_PATH` if you want to run a different released or local checkpoint directory. For example, to run a one-episode smoke test for `touchpick`:

```bash
HF_TOKEN=<your-huggingface-token-if-needed> \
FINETUNED_PATH=/path/to/local/checkpoint_or_run_dir \
NUM_EPISODES=1 EP_LENGTH=10 VERBOSE=False \
bash scripts/eval_touchpick.sh
```

The BSD-3 Clear License of this repository also applies to the model weights.

## Usage
The main evaluation entrypoint is [eval_scratchpad_vla.py](eval_scratchpad_vla.py). In practice, this repository is intended to be driven through the task-specific shell scripts in [scripts](scripts), which populate the Python path, resolve the checkpoint to evaluate, and configure rollout parameters before invoking the evaluator.

The provided scripts default to `VLA_VARIANT=sp` and currently cover the following tasks:
- `place_next_restore`
- `rotate_restore`
- `stack_topple`
- `swap`
- `touchpick`

To run the released `touchpick` SP evaluation path:
```bash
bash scripts/eval_touchpick.sh
```

By default, the task scripts evaluate `50` episodes and write outputs under `eval_runs/<task_name>/results/`.

Useful overrides for a short smoke test:
```bash
NUM_EPISODES=1 EP_LENGTH=10 VERBOSE=False \
bash scripts/eval_touchpick.sh
```

Evaluation dependency paths are centralized in [scripts/setup.sh](scripts/setup.sh). The task scripts source it automatically, and each path can be overridden through environment variables:
- `MANISKILL2_ROOT`
- `CLEVRSKILLS_ROOT`

The Python evaluator does not require additional user-configured environment variables; it sets `TOKENIZERS_PARALLELISM=false` internally to avoid tokenizer worker warnings.

Set `HF_TOKEN` in the environment if the required base model weights are not already available in the local Hugging Face cache.

## Repository Structure
- [docker](docker): Docker image, entrypoint, and runtime dependency files.
- [scripts](scripts): Shell entrypoints for each evaluation task.
- [eval_scratchpad_vla.py](eval_scratchpad_vla.py): Main rollout driver that loads checkpoints, instantiates the ClevrSkills environment, runs episodes, and records outputs.
- [paligemma_runner.py](paligemma_runner.py): PaliGemma inference wrapper with scratchpad-aware prompting and action generation.
- [eval_utils.py](eval_utils.py): Environment helpers, prompt/task utilities, and image preprocessing routines.
- [action_tokenizer.py](action_tokenizer.py): Action token discretization and decoding utilities used by the runner.
- [utils.py](utils.py): Checkpoint discovery and selection helpers used during evaluation.
- [eval_runs](eval_runs): Default output location for logs, trajectories, and evaluation artifacts.

## Reproducing Evaluations
The task-specific shell scripts are the intended reproduction surface for the released evaluations. Each command below runs one task at a selected checkpoint epoch.

```bash
# Place-next-to-restore
bash scripts/eval_place_next_restore.sh

# Stack-topple
bash scripts/eval_stack_topple.sh

# Swap
bash scripts/eval_swap.sh

# Touch-pick
bash scripts/eval_touchpick.sh

# Rotate-restore
bash scripts/eval_rotate_restore.sh
```

Run each task-specific script separately to reproduce the released evaluations.

## Troubleshooting
- If a task script cannot import ManiSkill2 or ClevrSkills, set `MANISKILL2_ROOT` and `CLEVRSKILLS_ROOT` to the local paths of the installed repositories before running the script.
- If model loading fails with an authentication error, set `HF_TOKEN` in your environment or make sure the required model weights are already available in the local Hugging Face cache.
- If rendering fails inside Docker, verify that the host has NVIDIA Container Toolkit installed and that the container was started with `--gpus=all` and sufficient shared memory.
- If a checkpoint path cannot be found, set `FINETUNED_PATH` to the checkpoint directory you want to evaluate.
- For any Vulkan / Sapien related issue, see [Vulkan / Sapien troubleshooting](https://github.com/Qualcomm-AI-research/ClevrSkills#troubleshooting).

### Docker permissions and host paths

If Docker fails before the container starts with an error like
`error while creating mount source path ... permission denied`, your current checkout is
under a host path the Docker daemon cannot traverse. In that case, use the code copied
into the image and mount only the Hugging Face cache:

```bash
mkdir -p "$HOME/.cache/huggingface"

docker run --rm -it \
  --gpus=all \
  --shm-size=32g \
  -v "$HOME/.cache/huggingface":/workspace/hf_cache \
  -w /workspace/scratchpad-vla \
  scratchpad-eval:latest \
  /bin/bash
```

If you need live editing from the host, copy the repository to a Docker-accessible local
path such as `/tmp/$USER/scratchpad-vla` and mount that path instead of the current checkout.

If your checkpoints are not already inside the container, mount the host directory that
contains them and set `FINETUNED_PATH` to the exact checkpoint directory. For example,
mount with `-v /host/path/to/checkpoints:/external_checkpoints:ro` and set
`FINETUNED_PATH=/external_checkpoints/`.

## Code formatting

The code was formatted using:

```bash
python -m black --safe --line-length 100 .
python -m isort --profile black .
```

## Citation
```bibtex
@inproceedings{haresh2026notestoself,
  title={Notes-to-Self: Scratchpad Augmented VLAs for Memory Dependent Manipulation Tasks},
  author={Haresh, Sanjay and Dijkman, Daniel and Bhattacharyya, Apratim and Memisevic, Roland},
  booktitle={2026 IEEE International Conference on Robotics and Automation (ICRA)},
  year={2026},
  url={https://arxiv.org/abs/2602.21013}
}
```
