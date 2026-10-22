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
from contextlib import nullcontext
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


class TestSSDRunnerValidation(unittest.TestCase):
    def test_both_runners_allowed_but_graphs_rejected(self):
        source = ast.parse((_ROOT / "vllm_ascend/ascend_config.py").read_text(encoding="utf-8"))
        config = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == "AscendConfig")
        method = next(node for node in config.body if getattr(node, "name", None) == "derive_and_validate")
        # Execute only the SSD guard, before unrelated model configuration.
        method.body = method.body[:2]
        namespace = {"VllmConfig": object, "AscendConfig": object}
        exec(compile(ast.Module(body=[method], type_ignores=[]), "ssd_runner_guard", "exec"), namespace)
        validate = namespace[method.name]
        for runner in (False, True):
            for eager in (False, True):
                with self.subTest(v2=runner, eager=eager):
                    value = SimpleNamespace(ssd_weight_offload={"path": "/ssd"})
                    runtime = SimpleNamespace(
                        use_v2_model_runner=runner, model_config=SimpleNamespace(enforce_eager=eager)
                    )
                    if eager:
                        validate(value, runtime)
                    else:
                        with self.assertRaisesRegex(ValueError, "enforce-eager"):
                            validate(value, runtime)


class TestMRV2SSDInstallation(unittest.TestCase):
    def setUp(self):
        self.installed = Mock()
        upstream = ModuleType("vllm.model_executor.offloader.base")
        upstream.BaseOffloader = object
        upstream.NoopOffloader = object
        upstream.set_offloader = self.installed
        ascend = ModuleType("vllm_ascend.model_executor.offloader.prefetch")
        ascend.AscendPrefetchOffloader = Mock()
        with patch.dict(
            "sys.modules",
            {
                "vllm.model_executor.offloader.base": upstream,
                "vllm_ascend.model_executor.offloader.prefetch": ascend,
            },
        ):
            self.module = load_source("ssd_factory_under_test", _SOURCE / "base.py")
        self.instance = Mock()
        self.module.create_offloader = Mock(return_value=self.instance)
        self.config = SimpleNamespace(
            additional_config={"ssd_weight_offload": {"path": "/ssd"}},
            offload_config=object(),
            model_config=SimpleNamespace(enforce_eager=True),
            lora_config=None,
            speculative_config=None,
            parallel_config=SimpleNamespace(enable_eplb=False),
        )

    def test_disabled_leaves_parent_offloader_unchanged(self):
        self.config.additional_config = {}
        self.assertIsNone(self.module.install_ssd_offloader(self.config))
        self.installed.assert_not_called()
        self.module.create_offloader.assert_not_called()

    def test_enabled_installs_instance_with_eager_configuration(self):
        self.assertIs(self.module.install_ssd_offloader(self.config), self.instance)
        self.installed.assert_called_once_with(self.instance)
        self.module.create_offloader.assert_called_once_with(
            self.config.offload_config, {"path": "/ssd"}, enforce_eager=True
        )

    def test_incompatible_features_rejected_before_installation(self):
        for name in ("lora_config", "speculative_config", "enable_eplb"):
            with self.subTest(feature=name):
                owner = self.config.parallel_config if name == "enable_eplb" else self.config
                previous = getattr(owner, name)
                setattr(owner, name, True)
                with self.assertRaisesRegex(ValueError, "does not support"):
                    self.module.install_ssd_offloader(self.config)
                setattr(owner, name, previous)
        self.installed.assert_not_called()

    def test_v2_parent_install_load_finalize_shutdown_order(self):
        # Execute actual V2 lifecycle methods with a fake upstream runner. The
        # initializer is narrowed to its parent/install block, avoiding unrelated
        # attention/backend setup. No hardware behavior is simulated here.
        source = ast.parse((_ROOT / "vllm_ascend/worker/v2/model_runner.py").read_text(encoding="utf-8"))
        runner = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == "NPUModelRunner")
        methods = {node.name: node for node in runner.body if isinstance(node, ast.FunctionDef)}
        initializer = methods["__init__"]
        initializer.body = [next(node for node in initializer.body if isinstance(node, ast.With))]
        test_class = ast.ClassDef(
            name="Runner",
            bases=[ast.Name(id="Parent", ctx=ast.Load())],
            keywords=[],
            body=[initializer, methods["load_model"], methods["shutdown"]],
            decorator_list=[],
        )
        events = []
        instance = self.instance
        instance.post_init.side_effect = lambda: events.append("post_init")
        instance.close.side_effect = lambda: events.append("close")

        class Parent:
            def __init__(self, config, device):
                events.append("parent_init")
                self.vllm_config = config
                self.compilation_config = SimpleNamespace(cudagraph_mode="none")
                self.model_state = SimpleNamespace()
                self.prompt_logprobs_worker = None

            def load_model(self, *args, **kwargs):
                events.append("model_constructed")
                instance.post_init()

            def shutdown(self):
                events.append("parent_shutdown")

        def install(config):
            events.append("install")
            return self.module.install_ssd_offloader(config)

        namespace = {
            "Parent": Parent,
            "VllmConfig": object,
            "torch": SimpleNamespace(device=object),
            "torch_cuda_wrapper": nullcontext,
            "install_ssd_offloader": install,
            "get_pd_dspark_aux_layer_ids": lambda config: [],
            "CUDAGraphMode": SimpleNamespace(NONE="none"),
            "C8KVBlockZeroer": type("Zeroer", (), {}),
        }
        tree = ast.fix_missing_locations(ast.Module(body=[test_class], type_ignores=[]))
        exec(compile(tree, "v2_ssd_lifecycle", "exec"), namespace)
        value = namespace["Runner"](self.config, "npu:0")
        value.load_model()
        value.shutdown()
        self.assertEqual(
            events, ["parent_init", "install", "model_constructed", "post_init", "close", "parent_shutdown"]
        )
        instance.post_init.assert_called_once()
        self.assertIsNone(value.ssd_weight_offloader)


if __name__ == "__main__":
    unittest.main()
