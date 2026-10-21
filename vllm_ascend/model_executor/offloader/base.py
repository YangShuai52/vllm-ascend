from typing import TYPE_CHECKING

from vllm.model_executor.offloader.base import BaseOffloader, NoopOffloader

from vllm_ascend.model_executor.offloader.prefetch import AscendPrefetchOffloader

if TYPE_CHECKING:
    from vllm.config import OffloadConfig


def create_offloader(
    offload_config: "OffloadConfig | None", ssd_config: dict | None = None, *, enforce_eager: bool = False
) -> BaseOffloader:
    """Create an Ascend-aware offloader while preserving vLLM defaults."""

    if ssd_config is not None:
        # Lazy import keeps the experimental backend out of the default path.
        from vllm_ascend.model_executor.offloader.ssd import AscendSSDOffloader

        if not enforce_eager:
            raise ValueError("SSD weight offload requires --enforce-eager")
        if offload_config is None or offload_config.prefetch.offload_group_size <= 0:
            raise ValueError("SSD weight offload requires --offload-group-size > 0")
        if offload_config.uva.cpu_offload_gb > 0:
            raise ValueError("SSD weight offload cannot be combined with --cpu-offload-gb")
        prefetch = offload_config.prefetch
        return AscendSSDOffloader(
            directory=ssd_config["path"],
            group_size=prefetch.offload_group_size,
            num_in_group=prefetch.offload_num_in_group,
            offload_params=prefetch.offload_params,
        )

    if offload_config is None:
        return NoopOffloader()

    backend = offload_config.offload_backend
    prefetch = offload_config.prefetch

    if backend == "prefetch":
        return AscendPrefetchOffloader(
            group_size=prefetch.offload_group_size,
            num_in_group=prefetch.offload_num_in_group,
            prefetch_step=prefetch.offload_prefetch_step,
            offload_params=prefetch.offload_params,
            mode="cpu",
        )
    return NoopOffloader()
