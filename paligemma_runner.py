# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
"""PaliGemma VLA runner with scratchpad-aware action prediction."""

import json
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor

from action_tokenizer import ActionTokenizer
from eval_utils import crop_and_resize_fn

DEFAULT_MAX_NEW_TOKENS = 250
SP_VARIANT = "sp"
Scratchpad = Tuple[Optional[str], str, str]


class PaliGemmaVLARunner:
    """Run PaliGemma checkpoints for scratchpad VLA evaluation."""

    def __init__(
        self,
        model_path: Union[str, Path],
        finetuned_path: Union[str, Path],
        vla_variant: str,
        do_sample: bool = False,
        custom_generate: bool = False,
    ) -> None:
        """Load a base model, processor, and checkpoint normalization stats."""
        self.model_path = str(model_path)
        self.finetuned_path = finetuned_path
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.vla_variant = vla_variant

        self.do_sample = do_sample
        self.custom_generate = custom_generate
        self.do_sample_initial_custom = True

        if "paligemma2" in self.model_path:
            self.processor: Any = PaliGemmaProcessor.from_pretrained(
                "google/paligemma2-3b-pt-224", trust_remote_code=True
            )
        else:
            self.processor: Any = PaliGemmaProcessor.from_pretrained(
                "google/paligemma-3b-pt-224", trust_remote_code=True
            )

        self.vla: Any = PaliGemmaForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            return_dict_in_generate=True,
            attn_implementation="eager",
        )
        self.vla = self.vla.to(self.device)
        self.vla.eval()

        finetuned_path = Path(self.finetuned_path)
        if finetuned_path.is_dir():
            try:
                with open(
                    finetuned_path / "dataset_statistics.json",
                    "r",
                    encoding="utf-8",
                ) as f:
                    self.norm_stats = json.load(f)
            except FileNotFoundError:
                with open(
                    finetuned_path.parent.parent / "dataset_statistics.json",
                    "r",
                    encoding="utf-8",
                ) as f:
                    self.norm_stats = json.load(f)
        else:
            raise FileNotFoundError(
                f"Fine-tuned checkpoint directory not found: {self.finetuned_path}"
            )

        self.aux_thoughts = []
        self.aux_positions = []
        self.sp_plan = []
        self.last_scratchpad = None
        self.last_thought_valid = True
        self.prev_actions = None
        self.action_tokenizer = ActionTokenizer(self.tokenizer)

    @property
    def tokenizer(self) -> Any:
        """Return the tokenizer attached to the loaded processor."""
        return getattr(self.processor, "tokenizer")

    def reset_aux(self) -> None:
        """Reset per-episode scratchpad and auxiliary trace state."""
        self.aux_thoughts = []
        self.aux_positions = []
        self.sp_plan = []
        self.last_scratchpad = None
        self.last_thought_valid = True
        self.prev_actions = None

    @staticmethod
    def _check_unnorm_key(
        norm_stats: Dict[str, Dict[str, Any]],
        unnorm_key: Optional[str],
    ) -> str:
        """Validate and return the dataset key for action unnormalization."""
        if unnorm_key is None:
            assert len(norm_stats) == 1, (
                f"Your model was trained on more than one dataset, "
                f"please pass a `unnorm_key` from the following options to "
                f"choose the statistics "
                f"used for un-normalizing actions: {norm_stats.keys()}"
            )
            unnorm_key = next(iter(norm_stats.keys()))

        assert unnorm_key in norm_stats, (
            f"The `unnorm_key` is not in the available dataset statistics, "
            f"please choose from: {norm_stats.keys()}"
        )
        return unnorm_key

    def get_action_dim(self, unnorm_key: Optional[str] = None) -> int:
        """Return the configured action dimensionality for a dataset key."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return len(self.norm_stats[unnorm_key]["action"]["q01"])

    def get_action_stats(self, unnorm_key: Optional[str] = None) -> Dict[str, Any]:
        """Return action normalization statistics for a dataset key."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return self.norm_stats[unnorm_key]["action"]

    def preprocess_image_prompt(
        self,
        image: np.ndarray,
        language_addendum: str,
        instruction: str,
        center_crop: bool,
    ) -> Tuple[Image.Image, str]:
        """Convert an observation image and instruction into model inputs."""
        image = Image.fromarray(image)
        image = image.convert("RGB")

        if center_crop:
            image = crop_and_resize_fn(image)

        prompt = f"What action should the robot take to {instruction.lower()}?"
        if language_addendum != "":
            prompt = language_addendum + "\n" + prompt

        if len(self.sp_plan) > 0:
            prompt = prompt + "\n" + f"<plan>{''.join(self.sp_plan)}</plan>"

        return image, prompt

    @torch.inference_mode()
    def predict_action(
        self,
        image: np.ndarray,
        language_addendum: str,
        instruction: str,
        unnorm_key: Optional[str],
        center_crop: bool = False,
        verbose: bool = False,
        use_oracle_thoughts: bool = False,
        ep_start: bool = False,
    ) -> np.ndarray:
        """Predict one environment action from an image and instruction."""
        if self.vla_variant != SP_VARIANT:
            raise NotImplementedError()
        if use_oracle_thoughts:
            raise NotImplementedError()

        image, prompt = self.preprocess_image_prompt(
            image, language_addendum, instruction, center_crop
        )
        inputs = self.processor(images=image, text="<image>" + prompt, return_tensors="pt").to(
            self.device
        )
        impromptu = "<plan>" if ep_start else "<think>"
        extra_inputs = self.tokenizer(impromptu, return_tensors="pt")
        inputs.data["input_ids"] = torch.cat(
            [inputs.data["input_ids"], extra_inputs.data["input_ids"].to(self.device)],
            dim=1,
        )
        inputs.data["attention_mask"] = torch.cat(
            [
                inputs.data["attention_mask"],
                extra_inputs.data["attention_mask"].to(self.device),
            ],
            dim=1,
        )

        action, _, full_output_tokens, scratchpad = self.predict_actions(
            inputs,
            unnorm_key=unnorm_key,
            ep_start=ep_start,
        )
        self.last_scratchpad = scratchpad
        output_decoded = self.tokenizer.decode(
            full_output_tokens.squeeze().tolist(), skip_special_tokens=True
        )
        self.aux_thoughts.append(output_decoded)

        if verbose:
            print(f"{self.vla_variant} System 2 output: {output_decoded}")
            if scratchpad is not None:
                print("PLAN: ", scratchpad[0])
                print("THOUGHT: ", scratchpad[1])
                print("ACT CLAUSE: ", scratchpad[2])

        if scratchpad is not None:
            plan, thought, act_clause = scratchpad
            if ep_start:
                self.sp_plan.append(plan)
            if "<done>" in act_clause:
                self.last_thought_valid = False
                self.sp_plan.append(thought)

        return action

    @torch.inference_mode()
    def predict_actions(
        self,
        model_inputs: Optional[Dict[str, torch.Tensor]] = None,
        unnorm_key: Optional[str] = None,
        ep_start: bool = False,
    ) -> Tuple[np.ndarray, None, torch.Tensor, Optional[Scratchpad]]:
        """Generate action tokens and decode them into environment actions."""
        norm_keys = list(self.norm_stats.keys())
        if len(norm_keys) == 1:
            unnorm_key = norm_keys[0]

        scratchpad = None
        if self.custom_generate:
            generated_ids, _, predicted_action_token_ids = self.custom_generate_actions(
                model_inputs, unnorm_key
            )
        else:
            try:
                generated_output = self.vla.generate(
                    **model_inputs,
                    max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
                    do_sample=self.do_sample,
                    use_cache=True,
                )
                generated_ids = generated_output["sequences"]
                predicted_action_token_ids = (
                    generated_ids[0, -(self.get_action_dim(unnorm_key) + 1) : -1].cpu().numpy()
                )

                all_valid = np.all(
                    predicted_action_token_ids >= self.action_tokenizer.action_token_begin_idx
                ) and np.all(
                    predicted_action_token_ids < self.action_tokenizer.action_token_end_idx
                )
                if not all_valid:
                    raise ValueError("Action not valid", predicted_action_token_ids)

                generated_tokens = self.tokenizer.decode(generated_ids[0])
                plan = None
                if ep_start:
                    plan = re.search(r"<plan>(.*?)</plan>", generated_tokens, re.DOTALL).group(1)
                thought = re.search(r"<think>(.*?)</think>", generated_tokens, re.DOTALL).group(1)
                act_clause = re.search(
                    re.escape("<act>") + r"(.*)", generated_tokens, re.DOTALL
                ).group(1)
                scratchpad = (plan, thought, act_clause)
            except ValueError:
                generated_ids, _, predicted_action_token_ids = self.custom_generate_actions(
                    model_inputs, unnorm_key
                )

        normalized_actions = self.action_tokenizer.decode_token_ids_to_actions(
            predicted_action_token_ids
        )
        action_norm_stats = self.get_action_stats(unnorm_key)
        mask = action_norm_stats.get(
            "mask",
            np.ones_like(action_norm_stats["q01"], dtype=bool),
        )
        action_high, action_low = (
            np.array(action_norm_stats["q99"]),
            np.array(action_norm_stats["q01"]),
        )
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )
        self.prev_actions = actions

        return actions, None, generated_ids, scratchpad

    def custom_generate_actions(
        self,
        model_inputs: Dict[str, torch.Tensor],
        unnorm_key: Optional[str],
        max_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    ) -> Tuple[torch.Tensor, Dict[str, Any], np.ndarray]:
        """Autoregressively generate text and action tokens without caching."""
        generated_tokens = 0
        generated_actions = 0
        policy_mode = False
        generated_output = {
            "sequences": model_inputs["input_ids"].tolist(),
            "scores": [],
        }
        eos_token = self.tokenizer.vocab[self.tokenizer.eos_token]
        action_token_begin_idx = self.action_tokenizer.action_token_begin_idx
        action_token_end_idx = self.action_tokenizer.action_token_end_idx
        last_generated_token = -1

        while last_generated_token != eos_token and generated_tokens < max_tokens:
            out = self.vla(**model_inputs)

            if not policy_mode and "<act>\n" in self.tokenizer.decode(
                generated_output["sequences"][0][-4:]
            ):
                policy_mode = True

            if policy_mode:
                if generated_actions < self.get_action_dim(unnorm_key):
                    action_logits = out["logits"][
                        0,
                        -1,
                        action_token_begin_idx:action_token_end_idx,
                    ]
                    if generated_actions == 0 and self.do_sample_initial_custom:
                        last_generated_token = (
                            torch.multinomial(
                                torch.softmax(action_logits, dim=-1),
                                num_samples=1,
                            )
                            + action_token_begin_idx
                        )
                    else:
                        last_generated_token = action_logits.argmax() + action_token_begin_idx
                    last_generated_token = last_generated_token.item()
                    generated_actions += 1
                else:
                    last_generated_token = eos_token
            else:
                if generated_tokens == 0 and self.do_sample_initial_custom:
                    last_generated_token = torch.multinomial(
                        torch.softmax(out["logits"][0, -1, :], dim=-1), num_samples=1
                    ).item()
                else:
                    last_generated_token = out["logits"][0, -1, :].argmax().item()

            generated_tokens += 1
            generated_output["sequences"][0].append(last_generated_token)
            model_inputs["input_ids"] = torch.cat(
                [
                    model_inputs["input_ids"],
                    torch.tensor([last_generated_token])
                    .long()
                    .reshape(1, 1)
                    .to(model_inputs["input_ids"].device),
                ],
                dim=1,
            )
            model_inputs["attention_mask"] = torch.cat(
                [
                    model_inputs["attention_mask"],
                    torch.tensor([1])
                    .long()
                    .reshape(1, 1)
                    .to(model_inputs["attention_mask"].device),
                ],
                dim=1,
            )

        generated_ids = torch.tensor(generated_output["sequences"]).long()
        predicted_action_token_ids = (
            generated_ids[0, -(self.get_action_dim(unnorm_key) + 1) : -1].cpu().numpy()
        )
        return generated_ids, generated_output, predicted_action_token_ids
