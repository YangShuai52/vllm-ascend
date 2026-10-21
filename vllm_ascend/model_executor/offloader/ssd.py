"""Experimental eager-only SSD layer streaming for Ascend, including 310P.

Reuses the CPU offloader's construction/loading and NZ format detection. The
startup loader still holds offloaded weights in RAM; only steady-state inference
uses two RAM slots and two HBM slots. Files are private to this worker/run and
are deliberately retained for inspection (no automatic directory deletion).
"""

import atexit
import json
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from vllm.logger import logger
from vllm.model_executor.offloader.prefetch import StaticBufferPool

from vllm_ascend.model_executor.offloader.prefetch import (
    AscendPrefetchOffloader,
    _format_static_buffers_for_nz,
)
from vllm_ascend.model_executor.offloader.ssd_io import read_exact_into

_SLOTS = 2
_ALIGNMENT = 64


class AscendSSDOffloader(AscendPrefetchOffloader):
    """Two-slot SSD/RAM/HBM experiment; homogeneous selected layers only.

    All scheduling and NPU submissions happen on the caller thread. A single
    I/O worker waits for a slot's H2D event before overwriting pinned RAM.
    Monotonic sequence numbers keep odd layer counts safe across forward calls.
    """

    def __init__(self, directory: str, **kwargs):
        super().__init__(prefetch_step=_SLOTS, mode="cpu", **kwargs)
        root = Path(directory).expanduser()
        root.mkdir(parents=True, exist_ok=True)
        self.directory = Path(tempfile.mkdtemp(prefix="ascend-weights-", dir=root))
        self.sequence = 0
        self.initialized = False
        self.executor = None
        self.reads = {}
        self.bytes_read = 0

    def _hook_module_forward(self, index, module):
        original_forward = module.forward

        def forward(*args, **kwargs):
            if not self.initialized:
                raise RuntimeError("SSD offloader post_init() has not completed")
            if index != self.sequence % len(self.module_offloaders):
                raise RuntimeError("SSD experiment requires sequential, non-reentrant decoder layer execution")
            slot = self.sequence % _SLOTS
            offloader = self.module_offloaders[index]
            # Rebind on every call: odd layer counts change slot parity between tokens.
            for info in self.infos[index]:
                parameter = offloader._param_offloaders[info.name]
                buffer = self.buffer_pool.get_buffer(info.name, info.shape, info.stride, info.dtype, slot)
                parameter._gpu_buffer = buffer
                parameter._param.data = buffer
            torch.npu.current_stream().wait_event(self.copy_done[slot])
            result = original_forward(*args, **kwargs)
            self.compute_done[slot].record(torch.npu.current_stream())
            self.compute_valid[slot] = True
            # Current compute is already queued: waiting for disk here can overlap it.
            self.sequence += 1
            self._upload(self.sequence)
            return result

        module.forward = forward

    def post_init(self):
        if self.initialized:
            raise RuntimeError("SSD offloader post_init() must only run once")
        if not self.module_offloaders:
            raise ValueError("SSD weight offload selected no decoder layers/parameters")
        self.copy_stream = torch.npu.Stream()
        self.device = self.module_offloaders[0].device
        self.infos = []
        for offloader in self.module_offloaders:
            offloader.sync_cpu_storage()
            self.infos.append(offloader.get_param_infos())
        signatures = [[(info.key, info.use_nz_buffer) for info in infos] for infos in self.infos]
        if any(signature != signatures[0] for signature in signatures[1:]):
            raise ValueError("SSD experiment requires identical selected parameter shapes/strides/formats per layer")
        self.buffer_pool = StaticBufferPool(self.infos[0], _SLOTS, self.device)
        for index, offloader in enumerate(self.module_offloaders):
            offloader.assign_buffer_slot(self.buffer_pool, index % _SLOTS)
        # Eager permits ND CPU to NZ HBM conversion on 310P. Do not disable NZ.
        _format_static_buffers_for_nz(
            self.module_offloaders, self.buffer_pool, [info for infos in self.infos for info in infos]
        )
        self.layout = []
        layer_bytes = 0
        for info in self.infos[0]:
            layer_bytes = (layer_bytes + _ALIGNMENT - 1) // _ALIGNMENT * _ALIGNMENT
            self.layout.append((layer_bytes, info.num_bytes))
            layer_bytes += info.num_bytes
        self.layer_bytes = layer_bytes
        manifest = []
        for index, offloader in enumerate(self.module_offloaders):
            path = self.directory / f"layer-{index:05d}.bin"
            with path.open("wb", buffering=0) as output:
                output.truncate(layer_bytes)
                for info, (offset, _) in zip(self.infos[index], self.layout):
                    parameter = offloader._param_offloaders[info.name]
                    data = parameter._cpu_storage.contiguous().reshape(-1).view(torch.uint8).numpy()
                    output.seek(offset)
                    view = memoryview(data)
                    written = 0
                    while written < len(view):
                        count = output.write(view[written:])
                        if not count:
                            raise OSError(f"Short write exporting SSD weights: {path}")
                        written += count
                    del view, data
                    # Parameters now point to HBM; release the full CPU backing.
                    parameter._cpu_storage = None
            manifest.append({"layer": index, "file": path.name, "bytes": layer_bytes})
        (self.directory / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        self.ram = [torch.empty(layer_bytes, dtype=torch.uint8, pin_memory=True) for _ in range(_SLOTS)]
        self.ram_views = [memoryview(buffer.numpy()) for buffer in self.ram]
        self.copy_done = [torch.npu.Event() for _ in range(_SLOTS)]
        self.compute_done = [torch.npu.Event() for _ in range(_SLOTS)]
        self.compute_valid = [False] * _SLOTS
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ssd-weight-read")
        atexit.register(self.close)
        # Static-buffer allocation/NZ conversion was queued on the compute stream.
        self.copy_stream.wait_stream(torch.npu.current_stream())
        self._queue_read(0)
        self._queue_read(1)
        self._upload(0)
        self.initialized = True
        logger.info(
            "SSD weight experiment: layers=%d RAM=%d bytes HBM pool=%d bytes files=%s; "
            "startup RAM and OS page cache are NOT bounded by the staging pool",
            len(self.module_offloaders),
            _SLOTS * layer_bytes,
            self.buffer_pool.total_bytes,
            self.directory,
        )

    def _queue_read(self, sequence, previous_copy=None):
        def read():
            if previous_copy is not None:
                torch.npu.set_device(self.device)
                previous_copy.synchronize()
            index = sequence % len(self.module_offloaders)
            read_exact_into(self.directory / f"layer-{index:05d}.bin", self.ram_views[sequence % _SLOTS])

        self.reads[sequence] = self.executor.submit(read)

    def _upload(self, sequence):
        self.reads.pop(sequence).result()
        slot = sequence % _SLOTS
        index = sequence % len(self.module_offloaders)
        if self.compute_valid[slot]:
            self.copy_stream.wait_event(self.compute_done[slot])
        with torch.npu.stream(self.copy_stream):
            for info, (offset, size) in zip(self.infos[index], self.layout):
                source = self.ram[slot][offset : offset + size].view(info.dtype).reshape(info.shape)
                target = self.buffer_pool.get_buffer(info.name, info.shape, info.stride, info.dtype, slot)
                target.copy_(source, non_blocking=True)
            # Fresh event per transfer: the I/O worker must not observe a re-recorded event.
            event = torch.npu.Event()
            event.record(self.copy_stream)
            self.copy_done[slot] = event
        self.bytes_read += self.layer_bytes
        self._queue_read(sequence + _SLOTS, event)

    def sync_prev_onload(self):
        torch.npu.current_stream().wait_stream(self.copy_stream)

    def join_after_forward(self):
        self.sync_prev_onload()

    def close(self):
        """Drain I/O before releasing buffers; retain exported files for inspection."""
        if self.executor is not None:
            self.executor.shutdown(wait=True)
            self.executor = None
