from typing import TYPE_CHECKING

from vllm.model_executor.offloader.base import BaseOffloader, NoopOffloader, set_offloader

from vllm_ascend.model_executor.offloader.prefetch import AscendPrefetchOffloader

if TYPE_CHECKING:
    from vllm.config import OffloadConfig, VllmConfig


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


def install_ssd_offloader(vllm_config: "VllmConfig") -> BaseOffloader | None:
    """Replace MRV2's default offloader before the model is constructed.

    The parent GPUModelRunner.load_model calls get_offloader().post_init() after
    weight processing. Do not call post_init again in the Ascend load wrapper.
    """
    ssd_config = (vllm_config.additional_config or {}).get("ssd_weight_offload")
    if ssd_config is None:
        return None
    if vllm_config.lora_config or vllm_config.speculative_config or vllm_config.parallel_config.enable_eplb:
        raise ValueError("SSD weight experiment does not support LoRA, speculative decoding or EPLB")
    offloader = create_offloader(
        vllm_config.offload_config, ssd_config, enforce_eager=vllm_config.model_config.enforce_eager
    )
    set_offloader(offloader)
    return offloader
