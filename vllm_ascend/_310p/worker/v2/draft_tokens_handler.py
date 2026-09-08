import numpy as np
import torch
from vllm.v1.outputs import DraftTokenIds
from vllm.v1.worker.gpu.input_batch import InputBatch


class Ascend310PDraftTokensHandler:
    """310P draft tokens handler: sync copy instead of CUDA stream/event."""

    def __init__(self, device: torch.device | None = None):
        self.device = device
        self.req_ids: list[str] = []
        self.draft_tokens_np: np.ndarray | None = None
        self.num_draft_tokens: int = 0

    def set_draft_tokens(
        self, input_batch: InputBatch, draft_tokens: torch.Tensor
    ) -> None:
        self.req_ids = input_batch.req_ids
        self.num_draft_tokens = draft_tokens.shape[1]
        # Synchronous copy to CPU for scheduler consumption.
        self.draft_tokens_np = draft_tokens.cpu().numpy()

    def get_draft_tokens(self) -> DraftTokenIds | None:
        if self.draft_tokens_np is not None:
            draft_token_ids = self.draft_tokens_np.tolist()
        else:
            draft_token_ids = [[-1] * self.num_draft_tokens for _ in self.req_ids]
        return DraftTokenIds(self.req_ids, draft_token_ids)
