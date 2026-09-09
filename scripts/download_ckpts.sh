#!/bin/bash
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

set -euo pipefail

BASE_URL="https://example.com/checkpoints"

cd "$(dirname "${BASH_SOURCE[0]}")/.."

mkdir -p checkpoints

wget "${BASE_URL}/place_next_restore.zip"
unzip -o place_next_restore.zip -d checkpoints
rm place_next_restore.zip

wget "${BASE_URL}/rotate_restore.zip"
unzip -o rotate_restore.zip -d checkpoints
rm rotate_restore.zip

wget "${BASE_URL}/stack_topple.zip"
unzip -o stack_topple.zip -d checkpoints
rm stack_topple.zip

wget "${BASE_URL}/swap.zip"
unzip -o swap.zip -d checkpoints
rm swap.zip

wget "${BASE_URL}/touch_pick.zip"
unzip -o touch_pick.zip -d checkpoints
rm touch_pick.zip
