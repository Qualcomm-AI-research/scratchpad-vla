# Copyright (c) 2024 Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, licensed under MIT 
# License is provided for attribution purposes only, Not a Contribution
 
# Changes from Qualcomm Technologies, Inc. are provided under the following license:
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear


"""Discretize continuous robot actions into tokenizer tokens."""

from typing import List, Union

import numpy as np
from transformers import (
    GemmaTokenizer,
    GemmaTokenizerFast,
    GPT2TokenizerFast,
    PreTrainedTokenizerBase,
)

DEFAULT_GEMMA_ACTION_TOKEN_END_IDX = 4000
DEFAULT_GPT2_ACTION_TOKEN_END_IDX = 356


class ActionTokenizer:
    """Map continuous robot actions to tokenizer tokens and back."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        bins: int = 256,
        min_action: float = -1.0,
        max_action: float = 1.0,
    ) -> None:
        """
        Discretizes continuous robot actions into N bins per dimension and maps
        to the least used tokens.

        NOTE =>> by default, assumes a BPE-style tokenizer akin to the
                 LlamaTokenizer, where *the least used tokens* appear at the
                 end of the vocabulary!

        Args:
            tokenizer: Base LLM/VLM tokenizer to extend.
            bins: Number of uniform bins for each continuous action value.
            min_action: Minimum action value used for clipping.
            max_action: Maximum action value used for clipping.
        """
        self.tokenizer, self.n_bins, self.min_action, self.max_action = (
            tokenizer,
            bins,
            min_action,
            max_action,
        )

        # Create Uniform Bins + Compute Bin Centers
        self.bins = np.linspace(min_action, max_action, self.n_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0

        self.tokenizer_len = self.tokenizer.vocab_size
        if isinstance(tokenizer, (GemmaTokenizerFast, GemmaTokenizer)):
            # these characters correspond to less used characters
            self.action_token_end_idx: int = int(
                self.tokenizer_len - DEFAULT_GEMMA_ACTION_TOKEN_END_IDX
            )
        elif isinstance(tokenizer, GPT2TokenizerFast):
            # these characters correspond to less used characters
            self.action_token_end_idx: int = DEFAULT_GPT2_ACTION_TOKEN_END_IDX
        else:
            # [Contract] Set "action_token_begin_idx" based on tokenizer size.
            #   =>> Assumes we overwrite the final `n_bins` vocabulary tokens.
            self.action_token_end_idx: int = int(self.tokenizer_len)
        # Note: end_idx is excluded, matching range(min, max).
        self.action_token_begin_idx: int = int(self.action_token_end_idx - self.n_bins)

    def tokenize_action(
        self,
        action: np.ndarray,
    ) -> Union[List[int], List[List[int]]]:
        """Clip and bin actions to the final ``n_bins`` vocabulary tokens."""
        action = np.clip(action, a_min=float(self.min_action), a_max=float(self.max_action))
        discretized_action = np.digitize(action, self.bins)

        # Handle single element vs. batch
        if len(discretized_action.shape) == 1:
            return list(self.action_token_end_idx - discretized_action)

        return (self.action_token_end_idx - discretized_action).tolist()

    def __call__(self, action: np.ndarray) -> Union[str, List[str]]:
        """Clip and bin actions to strings from the final vocabulary tokens."""
        action = np.clip(action, a_min=float(self.min_action), a_max=float(self.max_action))
        discretized_action = np.digitize(action, self.bins)

        # Handle single element vs. batch
        if len(discretized_action.shape) == 1:
            return self.tokenizer.decode(list(self.action_token_end_idx - discretized_action))

        return self.tokenizer.batch_decode(
            (self.action_token_end_idx - discretized_action).tolist()
        )

    def decode_token_ids_to_actions(self, action_token_ids: np.ndarray) -> np.ndarray:
        """
        Returns continuous actions for discrete action token IDs.

        NOTE =>> Because of the way the actions are discretized w.r.t. the bins
                 (and not the bin centers), the digitization returns bin indices
                 between [1, # bins], inclusive, when there are actually only
                 (# bins - 1) bin intervals.

                 Therefore, if the digitization returns the last possible index,
                 we map this to the last bin interval.

        EXAMPLE =>> Let's say self._bins has 256 values. Then self._bin_centers
                    has 255 values. Digitization returns indices between
                    [1, 256]. We subtract 1 from all indices so that they are
                    between [0, 255]. There is still one index (i==255) that
                    would cause an out-of-bounds error if used to index into
                    self._bin_centers. Therefore, if i==255, we subtract 1 from
                    it so that it just becomes the index of the last bin center.
                    We implement this simply via clipping between [0, 255 - 1].
        """
        discretized_actions = self.action_token_end_idx - action_token_ids
        discretized_actions = np.clip(
            discretized_actions - 1,
            a_min=0,
            a_max=self.bin_centers.shape[0] - 1,
        )

        return self.bin_centers[discretized_actions]

    @property
    def vocab_size(self) -> int:
        """Number of discrete action bins exposed by this tokenizer."""
        return self.n_bins
