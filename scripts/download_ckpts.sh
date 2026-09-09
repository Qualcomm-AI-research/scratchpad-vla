#!/usr/bin/env bash
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

set -euo pipefail

CHECKPOINT_DIR=${1-'./checkpoints/'}
BASE_URL="https://github.com/Qualcomm-AI-research/scratchpad-vla/releases/download/v1.0.0"

mkdir -p "${CHECKPOINT_DIR}"

wget -P "${CHECKPOINT_DIR}" "${BASE_URL}/place_next_restore.zip.part-00"
wget -P "${CHECKPOINT_DIR}" "${BASE_URL}/place_next_restore.zip.part-01"
wget -P "${CHECKPOINT_DIR}" "${BASE_URL}/place_next_restore.zip.part-02"
wget -P "${CHECKPOINT_DIR}" "${BASE_URL}/place_next_restore.zip.part-03"
cat "${CHECKPOINT_DIR}"/place_next_restore.zip.part-* > "${CHECKPOINT_DIR}/place_next_restore.zip"
unzip -o "${CHECKPOINT_DIR}/place_next_restore.zip" -d "${CHECKPOINT_DIR}"
rm "${CHECKPOINT_DIR}"/place_next_restore.zip*

wget -P "${CHECKPOINT_DIR}" "${BASE_URL}/rotate_restore.zip.part-00"
wget -P "${CHECKPOINT_DIR}" "${BASE_URL}/rotate_restore.zip.part-01"
wget -P "${CHECKPOINT_DIR}" "${BASE_URL}/rotate_restore.zip.part-02"
wget -P "${CHECKPOINT_DIR}" "${BASE_URL}/rotate_restore.zip.part-03"
cat "${CHECKPOINT_DIR}"/rotate_restore.zip.part-* > "${CHECKPOINT_DIR}/rotate_restore.zip"
unzip -o "${CHECKPOINT_DIR}/rotate_restore.zip" -d "${CHECKPOINT_DIR}"
rm "${CHECKPOINT_DIR}"/rotate_restore.zip*

wget -P "${CHECKPOINT_DIR}" "${BASE_URL}/stack_topple.zip.part-00"
wget -P "${CHECKPOINT_DIR}" "${BASE_URL}/stack_topple.zip.part-01"
wget -P "${CHECKPOINT_DIR}" "${BASE_URL}/stack_topple.zip.part-02"
wget -P "${CHECKPOINT_DIR}" "${BASE_URL}/stack_topple.zip.part-03"
cat "${CHECKPOINT_DIR}"/stack_topple.zip.part-* > "${CHECKPOINT_DIR}/stack_topple.zip"
unzip -o "${CHECKPOINT_DIR}/stack_topple.zip" -d "${CHECKPOINT_DIR}"
rm "${CHECKPOINT_DIR}"/stack_topple.zip*

wget -P "${CHECKPOINT_DIR}" "${BASE_URL}/swap.zip.part-00"
wget -P "${CHECKPOINT_DIR}" "${BASE_URL}/swap.zip.part-01"
wget -P "${CHECKPOINT_DIR}" "${BASE_URL}/swap.zip.part-02"
wget -P "${CHECKPOINT_DIR}" "${BASE_URL}/swap.zip.part-03"
cat "${CHECKPOINT_DIR}"/swap.zip.part-* > "${CHECKPOINT_DIR}/swap.zip"
unzip -o "${CHECKPOINT_DIR}/swap.zip" -d "${CHECKPOINT_DIR}"
rm "${CHECKPOINT_DIR}"/swap.zip*

wget -P "${CHECKPOINT_DIR}" "${BASE_URL}/touch_pick.zip.part-00"
wget -P "${CHECKPOINT_DIR}" "${BASE_URL}/touch_pick.zip.part-01"
wget -P "${CHECKPOINT_DIR}" "${BASE_URL}/touch_pick.zip.part-02"
wget -P "${CHECKPOINT_DIR}" "${BASE_URL}/touch_pick.zip.part-03"
cat "${CHECKPOINT_DIR}"/touch_pick.zip.part-* > "${CHECKPOINT_DIR}/touch_pick.zip"
unzip -o "${CHECKPOINT_DIR}/touch_pick.zip" -d "${CHECKPOINT_DIR}"
rm "${CHECKPOINT_DIR}"/touch_pick.zip*
