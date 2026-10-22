"""Dependency-free I/O and scheduling checks; runnable with Python unittest.

These checks do not validate Ascend stream semantics or NZ conversion.
"""

import ast
import importlib.util
import io
import os
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

_ROOT = Path(__file__).resolve().parents[3]
_SOURCE = _ROOT / "vllm_ascend/model_executor/offloader"


def load_source(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestSSDIO(unittest.TestCase):
    def setUp(self):
        self.module = load_source("ssd_io_under_test", _SOURCE / "ssd_io.py")

    def test_reads_existing_buffer_and_reports_truncation(self):
        descriptor, filename = tempfile.mkstemp(prefix="ssd-io-test-", dir=_ROOT)
        os.close(descriptor)
        path = Path(filename)
        try:
            path.write_bytes(b"weights")
            destination = bytearray(7)
            self.module.read_exact_into(path, memoryview(destination))
            self.assertEqual(destination, b"weights")
            with self.assertRaisesRegex(EOFError, "read 7/8 bytes"):
                self.module.read_exact_into(path, memoryview(bytearray(8)))
        finally:
            path.unlink()

    def test_partial_reads_are_retried(self):
        class PartialReader(io.BytesIO):
            def readinto(self, destination):
                return super().readinto(destination[:2])

        path = Mock()
        path.open.return_value = PartialReader(b"abcdef")
        destination = bytearray(6)
        self.module.read_exact_into(path, memoryview(destination))
        self.assertEqual(destination, b"abcdef")


class TestSSDReadScheduling(unittest.TestCase):
    def setUp(self):
        self.calls = []
        torch = ModuleType("torch")
        torch.npu = SimpleNamespace(
            set_device=lambda device: self.calls.append("device"), current_stream=Mock(return_value=Mock())
        )
        upstream = ModuleType("vllm.model_executor.offloader.prefetch")
        upstream.StaticBufferPool = Mock()
        ascend = ModuleType("vllm_ascend.model_executor.offloader.prefetch")
        ascend.AscendPrefetchOffloader = object
        ascend._format_static_buffers_for_nz = Mock()
        logger = ModuleType("vllm.logger")
        logger.logger = Mock()
        helper = load_source("ssd_io_under_test", _SOURCE / "ssd_io.py")
        with patch.dict(
            "sys.modules",
            {
                "torch": torch,
                "vllm.logger": logger,
                "vllm.model_executor.offloader.prefetch": upstream,
                "vllm_ascend.model_executor.offloader.prefetch": ascend,
                "vllm_ascend.model_executor.offloader.ssd_io": helper,
            },
        ):
            self.module = load_source("ssd_under_test", _SOURCE / "ssd.py")

    def test_ram_reuse_waits_for_h2d_and_wraps_odd_layer_count(self):
        offloader = self.module.AscendSSDOffloader.__new__(self.module.AscendSSDOffloader)
        offloader.device = "npu:0"
        offloader.directory = Path("unused")
        offloader.module_offloaders = [None] * 3
        offloader.ram_views = ["slot0", "slot1"]
        offloader.reads = {}
        event = SimpleNamespace(synchronize=lambda: self.calls.append("h2d-complete"))
        self.module.read_exact_into = lambda path, slot: self.calls.append((path.name, slot))
        with ThreadPoolExecutor(max_workers=1) as executor:
            offloader.executor = executor
            for sequence in range(8):
                offloader._queue_read(sequence, event)
                offloader.reads.pop(sequence).result()
        for sequence in range(8):
            self.assertEqual(
                self.calls[sequence * 3 : sequence * 3 + 3],
                ["device", "h2d-complete", (f"layer-{sequence % 3:05d}.bin", f"slot{sequence % 2}")],
            )

    def test_disk_error_propagates_before_h2d(self):
        offloader = self.module.AscendSSDOffloader.__new__(self.module.AscendSSDOffloader)
        with ThreadPoolExecutor(max_workers=1) as executor:

            def fail():
                raise EOFError("truncated weights")

            offloader.reads = {0: executor.submit(fail)}
            with self.assertRaisesRegex(EOFError, "truncated weights"):
                offloader._upload(0)

    def test_forward_rebinds_slots_across_single_even_and_odd_layer_counts(self):
        for count in (1, 2, 3, 5):
            with self.subTest(layers=count):
                offloader = self.module.AscendSSDOffloader.__new__(self.module.AscendSSDOffloader)
                offloader.initialized = True
                offloader.sequence = 0
                offloader.copy_done = [Mock(), Mock()]
                offloader.compute_done = [Mock(), Mock()]
                offloader.compute_valid = [False, False]
                offloader.buffer_pool = Mock()
                offloader.buffer_pool.get_buffer.side_effect = lambda *args: f"hbm-slot-{args[-1]}"
                offloader._upload = Mock()
                info = SimpleNamespace(name="weight", shape=(1,), stride=(1,), dtype="float")
                offloader.infos = [[info] for _ in range(count)]
                parameters = [SimpleNamespace(_param=SimpleNamespace(data=None)) for _ in range(count)]
                offloader.module_offloaders = [
                    SimpleNamespace(_param_offloaders={"weight": parameter}) for parameter in parameters
                ]
                seen = []
                modules = []
                for index in range(count):

                    def original(hidden_states, index=index, seen=seen, parameters=parameters):
                        seen.append((index, parameters[index]._param.data))
                        return hidden_states

                    module = SimpleNamespace(forward=original)
                    offloader._hook_module_forward(index, module)
                    modules.append(module)
                for sequence in range(count * 3):
                    self.assertEqual(modules[sequence % count].forward("hidden"), "hidden")
                    self.assertEqual(seen[-1], (sequence % count, f"hbm-slot-{sequence % 2}"))
                self.assertEqual(offloader.sequence, count * 3)
                self.assertEqual(offloader._upload.call_count, count * 3)

    def test_out_of_order_layer_fails_before_compute(self):
        offloader = self.module.AscendSSDOffloader.__new__(self.module.AscendSSDOffloader)
        offloader.initialized = True
        offloader.sequence = 0
        offloader.module_offloaders = [None, None]
        module = SimpleNamespace(forward=Mock())
        original = module.forward
        offloader._hook_module_forward(1, module)
        with self.assertRaisesRegex(RuntimeError, "sequential"):
            module.forward("hidden")
        original.assert_not_called()


class TestSSDConfiguration(unittest.TestCase):
    def test_registered_field_and_path_validation(self):
        # Exercise the real validator body without importing the NPU-dependent
        # module. Full pydantic construction still needs the target environment.
        source = ast.parse((_ROOT / "vllm_ascend/ascend_config.py").read_text(encoding="utf-8"))
        config = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == "AscendConfig")
        self.assertTrue(
            any(
                isinstance(node, ast.AnnAssign) and getattr(node.target, "id", None) == "ssd_weight_offload"
                for node in config.body
            )
        )
        validator = next(node for node in config.body if getattr(node, "name", None) == "_validate_ssd_weight_offload")
        validator.decorator_list = []
        namespace = {}
        exec(compile(ast.Module(body=[validator], type_ignores=[]), "ssd_config_validator", "exec"), namespace)
        validate = namespace[validator.name]
        self.assertIsNone(validate(None, None))
        value = {"path": "/home/data/offload"}
        self.assertEqual(validate(None, value), value)
        for invalid in ({}, {"path": " "}, {"path": "x", "typo": "y"}):
            with self.subTest(value=invalid), self.assertRaisesRegex(ValueError, "non-empty"):
                validate(None, invalid)


if __name__ == "__main__":
    unittest.main()
