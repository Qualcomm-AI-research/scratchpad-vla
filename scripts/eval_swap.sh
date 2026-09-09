#!/bin/bash
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/scripts/setup.sh"

PYTHON_BIN="${PYTHON_BIN:-/opt/conda/bin/python}"

EXTRA_PYTHONPATH="${EXTRA_PYTHONPATH:-}"
export PYTHONPATH="${REPO_ROOT}:${MANISKILL2_ROOT}:${CLEVRSKILLS_ROOT}${EXTRA_PYTHONPATH:+:${EXTRA_PYTHONPATH}}:${PYTHONPATH:-}"

VLA_VARIANT="${VLA_VARIANT:-sp}"
EVAL_EPOCH="${1:-${EVAL_EPOCH:-7}}"
NUM_EPISODES="${NUM_EPISODES:-50}"
EP_LENGTH="${EP_LENGTH:-250}"
START_SEED="${START_SEED:-30000}"
VERBOSE="${VERBOSE:-True}"
FINETUNED_PATH="${FINETUNED_PATH:-${REPO_ROOT}/checkpoints}/swap/checkpoint"
RECORD_DIR="${RECORD_DIR:-eval_runs/swap/results/}"

"${PYTHON_BIN}" eval_scratchpad_vla.py \
            --vla_variant "${VLA_VARIANT}" --start_seed "${START_SEED}" --record_dir "${RECORD_DIR}" \
            --num_episodes "${NUM_EPISODES}" --ep_length "${EP_LENGTH}" --task_description swap --num_objects 2 --verbose="${VERBOSE}" \
            --finetuned_path "${FINETUNED_PATH}" \
            --eval_epoch "${EVAL_EPOCH}"
