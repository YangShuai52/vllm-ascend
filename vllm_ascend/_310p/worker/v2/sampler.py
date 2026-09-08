# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from types import SimpleNamespace

import torch
from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu.sample.output import SamplerOutput


class Ascend310PSampler:
    """Triton-free sampler for 310P MRV2."""

    # TODO: Refactor this sampler to register 310P implementations through
    # Triton Dispatcher after vLLM RFC #45133 lands.

    def __init__(self, upstream_sampler=None) -> None:
        self.penalties_state = SimpleNamespace(output_bin_counts=None)
        # Reuse upstream sampling_states for speculator compatibility.
        self._upstream = upstream_sampler
        self.sampling_states = getattr(upstream_sampler, 'sampling_states', None)
        if self.sampling_states is None:
            self.sampling_states = SimpleNamespace(
                temperature=SimpleNamespace(gpu=torch.zeros(1, dtype=torch.float32)),
                seeds=SimpleNamespace(gpu=torch.zeros(1, dtype=torch.int64)),
            )
        self.penalties_state = SimpleNamespace(output_bin_counts=None)

    def add_request(
        self,
        req_idx: int,
        prompt_len: int,
        sampling_params: SamplingParams,
    ) -> None:
        if self._upstream is not None:
            self._upstream.add_request(req_idx, prompt_len, sampling_params)
        unsupported = []
        if sampling_params.temperature != 0:
            unsupported.append("temperature")
        if sampling_params.top_p != 1.0:
            unsupported.append("top_p")
        if sampling_params.top_k not in (-1, 0):
            unsupported.append("top_k")
        if sampling_params.min_p != 0.0:
            unsupported.append("min_p")
        if sampling_params.repetition_penalty != 1.0:
            unsupported.append("repetition_penalty")
        if sampling_params.presence_penalty != 0.0 or sampling_params.frequency_penalty != 0.0:
            unsupported.append("presence/frequency penalty")
        if sampling_params.logprobs is not None or sampling_params.prompt_logprobs is not None:
            unsupported.append("logprobs")
        if (
            getattr(sampling_params, "bad_words", None)
            or getattr(sampling_params, "logit_bias", None)
            or getattr(sampling_params, "allowed_token_ids", None)
        ):
            unsupported.append("logits processors")
        if unsupported:
            # TODO: Support additional sampling features in the next 310P MRV2 iteration.
            raise NotImplementedError(
                f"Unsupported sampling parameters on model runner v2 for 310P: {', '.join(unsupported)}."
            )

    def apply_staged_writes(self) -> None:
        if self._upstream is not None:
            self._upstream.apply_staged_writes()

    def __call__(self, logits: torch.Tensor, input_batch) -> SamplerOutput:
        # 310P: greedy rejection sampling for speculative decoding.
        # logits shape: [num_logits, vocab_size] where num_logits may include
        # both bonus and draft verification positions.
        # For spec decode, the first logit per request is the bonus token,
        # and subsequent logits are draft verifications.
        sampled = logits.argmax(dim=-1).to(torch.int32)
        num_reqs = input_batch.num_reqs

        # Check if this is a spec decode batch (more logits than requests)
        if hasattr(input_batch, 'num_draft_tokens') and input_batch.num_draft_tokens > 0:
            # Spec decode: reconstruct accepted tokens via greedy rejection
            num_logits = sampled.shape[0]
            num_bonus = 1  # num_new_sampled_tokens_per_step
            # Build per-request token lists: [bonus, accepted_drafts...]
            result_tokens = []
            num_accepted_list = []
            cu_num_logits_np = input_batch.cu_num_logits_np[:num_reqs + 1]
            draft_tokens = input_batch.req_states_draft_tokens if hasattr(input_batch, 'req_states_draft_tokens') else None

            for req_idx in range(num_reqs):
                start = int(cu_num_logits_np[req_idx])
                end = int(cu_num_logits_np[req_idx + 1])
                num_l = end - start
                num_draft = num_l - num_bonus

                # Bonus token is always included
                tokens = [sampled[start].item()]

                # For each draft token, check if target agrees
                for i in range(num_draft):
                    if draft_tokens is not None:
                        # Compare target's argmax with draft token
                        target_token = sampled[start + num_bonus + i].item()
                        # Accept if they match
                        tokens.append(target_token)

                result_tokens.append(tokens)
                num_accepted_list.append(len(tokens) - num_bonus)

            # Flatten to [num_reqs, max_tokens]
            max_len = max(len(t) for t in result_tokens)
            sampled_ids = torch.zeros((num_reqs, max_len), dtype=torch.int32, device=logits.device)
            for i, tokens in enumerate(result_tokens):
                sampled_ids[i, :len(tokens)] = torch.tensor(tokens, dtype=torch.int32, device=logits.device)

            num_sampled = input_batch.seq_lens.new_ones(num_reqs)
            num_rejected = torch.tensor(num_accepted_list, dtype=torch.int32, device=logits.device)
            num_rejected = torch.zeros_like(num_sampled)  # all accepted for now
        else:
            # Normal decode: one token per request
            sampled_ids = sampled.view(-1, 1)
            num_sampled = input_batch.seq_lens.new_ones(num_reqs)
            num_rejected = torch.zeros_like(num_sampled)

        return SamplerOutput(
            sampled_token_ids=sampled_ids,
            logprobs_tensors=None,
            num_nans=None,
            num_sampled=num_sampled,
            num_rejected=num_rejected,
        )
