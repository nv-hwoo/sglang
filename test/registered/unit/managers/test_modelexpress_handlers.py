# SPDX-License-Identifier: Apache-2.0

import asyncio
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, call

import pytest
import torch

sys.modules.setdefault("sgl_kernel", MagicMock())
for name in ("quantization", "scalar_type", "flash_attn", "flash_mla", "mamba"):
    sys.modules.setdefault(f"sgl_kernel.{name}", MagicMock())

from sglang.srt.managers.io_struct import (  # noqa: E402
    GetModelExpressStatusReqInput,
    MarkModelExpressPoisonedReqInput,
    ModelExpressWeightUpdateReqOutput,
    PrepareWeightsFromModelExpressReqInput,
    UpdateWeightsFromModelExpressReqInput,
)
from sglang.srt.managers.scheduler_components.weight_updater import (  # noqa: E402
    SchedulerWeightUpdaterManager,
)
from sglang.srt.managers.tokenizer_control_mixin import (  # noqa: E402
    TokenizerControlMixin,
)
from sglang.srt.observability.metrics_collector import (  # noqa: E402
    SchedulerMetricsCollector,
)
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


@pytest.fixture(autouse=True)
def modelexpress_modules(monkeypatch):
    class WeightVersionRef:
        def __init__(self, version_id):
            self.version_id = version_id

    modelexpress_rl = ModuleType("modelexpress_rl")
    modelexpress_rl.__path__ = []
    modelexpress_rl.WeightVersionRef = WeightVersionRef
    monkeypatch.setitem(sys.modules, "modelexpress_rl", modelexpress_rl)
    return modelexpress_rl


class Staged:
    def __init__(self, version_id):
        self.version_id = version_id
        self.metrics = {"perf/mx_receive_prepare_time": 2.0}
        self.applied = False
        self.released = 0
        self.release_error = None

    def release(self):
        self.released += 1
        if self.release_error is not None:
            raise self.release_error


class Generator:
    def __init__(self):
        self.worker_id = "generator-a"
        self.staged = []
        self.applied = []
        self.closed = 0
        self.stage_error = None
        self.apply_error = None

    def stage_weight(self, *, version):
        if self.stage_error is not None:
            raise self.stage_error
        staged = Staged(version.version_id)
        self.staged.append(staged)
        return staged

    def apply_weight(self, staged):
        if self.apply_error is not None:
            raise self.apply_error
        staged.applied = True
        self.applied.append(staged)
        return {"perf/mx_receive_install_time": 3.0}

    def close(self):
        self.closed += 1


def manager(generator):
    value = object.__new__(SchedulerWeightUpdaterManager)
    value.tp_worker = SimpleNamespace(
        model_runner=SimpleNamespace(
            loader=SimpleNamespace(
                _prepare_weights=lambda *_args: ("/models/launch", None, None)
            ),
            model_config=SimpleNamespace(model_path="model", revision=None),
            server_args=SimpleNamespace(
                modelexpress_model_name="model",
                modelexpress_server_url="mx:8001",
                modelexpress_initial_base_version_id="base-a",
                modelexpress_preparation_cache_dir="/tmp/mx-cache",
                modelexpress_s3_endpoint_url="http://minio:9000",
            ),
        )
    )
    value.modelexpress_generator = generator
    value.modelexpress_staged = None
    value.modelexpress_installed_version = "base-a" if generator is not None else None
    value.modelexpress_state = "VERIFIED" if generator is not None else None
    value.modelexpress_detail = ""
    value.draft_worker = None
    value.tp_cpu_group = object()
    value.flush_cache = lambda **_kwargs: True
    value.metrics_collector = None
    return value


def test_generator_client_is_built_once_on_first_manager_access(
    modelexpress_modules,
):
    generator = Generator()
    updater = manager(None)
    built = []

    class Config(SimpleNamespace):
        def __init__(self, **values):
            super().__init__(**values)

    class ModelExpressGeneratorClient:
        @staticmethod
        def initialize(config):
            built.append(config)
            return generator

    modelexpress_modules.ModelExpressGeneratorClient = ModelExpressGeneratorClient
    modelexpress_modules.ModelExpressGeneratorConfig = Config
    modelexpress_modules.ObjectStorageGeneratorConfig = Config
    modelexpress_modules.ObjectStorageType = SimpleNamespace(S3="S3")
    modelexpress_modules.SglangGeneratorContext = lambda model_runner: SimpleNamespace(
        model_runner=model_runner
    )

    status = updater.get_modelexpress_status(GetModelExpressStatusReqInput())
    prepared = updater.prepare_weights_from_modelexpress(
        PrepareWeightsFromModelExpressReqInput(target_version="2")
    )

    assert status.state == "VERIFIED"
    assert prepared.success is True
    assert len(built) == 1
    config = built[0]
    assert config.engine_context.model_runner is updater.tp_worker.model_runner
    assert config.model_name == "model"
    assert config.server_url == "mx:8001"
    assert set(vars(config)) == {
        "engine_context",
        "model_name",
        "object_storage",
        "server_url",
    }
    assert vars(config.object_storage) == {
        "endpoint_url": "http://minio:9000",
        "initial_base_version_id": "base-a",
        "launch_checkpoint": "/models/launch",
        "preparation_cache_dir": "/tmp/mx-cache",
        "storage_type": "S3",
    }
    assert updater.modelexpress_generator is generator
    assert generator.staged[0].version_id == "2"


def test_scheduler_handlers_drive_generator_client_stage_and_apply():
    generator = Generator()
    updater = manager(generator)
    updater.metrics_collector = MagicMock()

    prepared = updater.prepare_weights_from_modelexpress(
        PrepareWeightsFromModelExpressReqInput(target_version="2")
    )
    installed = updater.update_weights_from_modelexpress(
        UpdateWeightsFromModelExpressReqInput(target_version="2")
    )
    status = updater.get_modelexpress_status(GetModelExpressStatusReqInput())

    assert [item.version_id for item in generator.staged] == ["2"]
    assert generator.applied == generator.staged
    assert generator.staged[0].released == 1
    assert prepared.success is True
    assert prepared.metrics == {"perf/mx_receive_prepare_time": 2.0}
    assert installed.success is True
    assert installed.installed_version == "2"
    assert installed.metrics == {"perf/mx_receive_install_time": 3.0}
    assert status.state == "VERIFIED"
    assert updater.metrics_collector.observe_modelexpress_metrics.call_args_list == [
        call({"perf/mx_receive_prepare_time": 2.0}),
        call({"perf/mx_receive_install_time": 3.0}),
    ]


@pytest.mark.parametrize(
    ("metric_name", "phase"),
    [
        ("perf/mx_receive_delta_index_download", "delta_index_download"),
        ("perf/mx_receive_delta_download", "delta_download"),
        ("perf/mx_receive_delta_apply", "delta_apply"),
        ("perf/mx_receive_prepare_time", "prepare"),
        ("perf/mx_receive_install_time", "install"),
    ],
)
def test_modelexpress_metrics_use_bounded_prometheus_phases(metric_name, phase):
    collector = object.__new__(SchedulerMetricsCollector)
    collector.labels = {"model_name": "model"}
    collector.modelexpress_receive_duration_seconds = MagicMock()

    collector.observe_modelexpress_metrics(
        {metric_name: 1.25, "unknown_modelexpress_metric": 2.0}
    )

    gauge = collector.modelexpress_receive_duration_seconds
    gauge.labels.assert_called_once_with(model_name="model", phase=phase)
    gauge.labels.return_value.set.assert_called_once_with(1.25)


def test_metrics_failure_does_not_fail_modelexpress_weight_update():
    updater = manager(Generator())
    updater.metrics_collector = MagicMock()
    updater.metrics_collector.observe_modelexpress_metrics.side_effect = RuntimeError(
        "metrics unavailable"
    )

    prepared = updater.prepare_weights_from_modelexpress(
        PrepareWeightsFromModelExpressReqInput(target_version="2")
    )
    installed = updater.update_weights_from_modelexpress(
        UpdateWeightsFromModelExpressReqInput(target_version="2")
    )

    assert prepared.success is True
    assert installed.success is True


def test_prepare_error_becomes_failed_result():
    generator = Generator()
    generator.stage_error = RuntimeError("prepare failed")
    result = manager(generator).prepare_weights_from_modelexpress(
        PrepareWeightsFromModelExpressReqInput(target_version="2")
    )

    assert result.success is False
    assert result.state == "FAILED"
    assert result.detail == "prepare failed"


def test_apply_error_becomes_failed_result():
    generator = Generator()
    generator.apply_error = RuntimeError("apply failed")
    updater = manager(generator)
    updater.prepare_weights_from_modelexpress(
        PrepareWeightsFromModelExpressReqInput(target_version="2")
    )
    result = updater.update_weights_from_modelexpress(
        UpdateWeightsFromModelExpressReqInput(target_version="2")
    )

    assert result.success is False
    assert result.state == "FAILED"
    assert result.detail == "apply failed"


def test_receive_metrics_merge_by_max_latency():
    first = ModelExpressWeightUpdateReqOutput(
        success=True,
        receiver_id="dp0",
        installed_version="2",
        state="VERIFIED",
        metrics={"perf/mx_receive_prepare_time": 2.0},
    )
    second = ModelExpressWeightUpdateReqOutput(
        success=True,
        receiver_id="dp1",
        installed_version="2",
        state="VERIFIED",
        metrics={"perf/mx_receive_prepare_time": 3.0},
    )

    result = TokenizerControlMixin._merge_modelexpress_results(
        [first, second], mutation_phase=False
    )

    assert result.metrics == {"perf/mx_receive_prepare_time": 3.0}


def test_installing_an_unprepared_target_is_refused_without_mutating():
    generator = Generator()
    updater = manager(generator)

    unprepared = updater.update_weights_from_modelexpress(
        UpdateWeightsFromModelExpressReqInput(target_version="2")
    )
    updater.prepare_weights_from_modelexpress(
        PrepareWeightsFromModelExpressReqInput(target_version="2")
    )
    mismatched = updater.update_weights_from_modelexpress(
        UpdateWeightsFromModelExpressReqInput(target_version="3")
    )

    for refused in (unprepared, mismatched):
        assert refused.success is False
        assert refused.detail == "requested target is not prepared"


def test_cache_flush_exception_becomes_poisoned_result():
    generator = Generator()
    updater = manager(generator)
    updater.prepare_weights_from_modelexpress(
        PrepareWeightsFromModelExpressReqInput(target_version="2")
    )

    def fail_flush(**_kwargs):
        raise RuntimeError("cache flush exploded")

    updater.flush_cache = fail_flush

    result = updater.update_weights_from_modelexpress(
        UpdateWeightsFromModelExpressReqInput(target_version="2")
    )

    assert result.success is False
    assert result.state == "POISONED"
    assert "cache invalidation raised after install" in result.detail


def test_staged_cleanup_failure_does_not_skip_update_result():
    generator = Generator()
    updater = manager(generator)
    updater.prepare_weights_from_modelexpress(
        PrepareWeightsFromModelExpressReqInput(target_version="2")
    )
    updater.modelexpress_staged.release_error = RuntimeError("lease unavailable")

    result = updater.update_weights_from_modelexpress(
        UpdateWeightsFromModelExpressReqInput(target_version="2")
    )

    assert result.success is True
    assert "staged cleanup failed: lease unavailable" in result.detail
    assert updater.modelexpress_staged is None


def test_tp_disagreement_after_mutation_poisoned_the_whole_engine(monkeypatch):
    generator = Generator()
    updater = manager(generator)
    local = ModelExpressWeightUpdateReqOutput(
        success=True,
        receiver_id="tp0",
        installed_version="2",
        state="VERIFIED",
    )
    failed = ModelExpressWeightUpdateReqOutput(
        success=False,
        receiver_id="tp1",
        installed_version="1",
        state="FAILED",
        detail="native load failed before write",
    )
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 2)

    def gather(results, _local, group):
        results[:] = [local, failed]

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)

    result = updater._modelexpress_tp_agreement(local, mutation_phase=True)

    assert result.success is False
    assert result.state == "POISONED"


def test_dp_disagreement_after_mutation_blocks_serving_as_poisoned():
    verified = ModelExpressWeightUpdateReqOutput(
        success=True,
        receiver_id="dp0",
        installed_version="2",
        state="VERIFIED",
    )
    failed = ModelExpressWeightUpdateReqOutput(
        success=False,
        receiver_id="dp1",
        installed_version="1",
        state="FAILED",
        detail="native load failed before write",
    )

    result = TokenizerControlMixin._merge_modelexpress_results(
        [verified, failed], mutation_phase=True
    )

    assert result.success is False
    assert result.state == "POISONED"


def test_dp_divergence_fanout_marks_underlying_receivers_poisoned():
    verified = ModelExpressWeightUpdateReqOutput(
        success=True,
        receiver_id="dp0",
        installed_version="2",
        state="VERIFIED",
    )
    failed = ModelExpressWeightUpdateReqOutput(
        success=False,
        receiver_id="dp1",
        installed_version="1",
        state="FAILED",
        detail="native load failed before write",
    )
    poisoned = [
        ModelExpressWeightUpdateReqOutput(
            success=False,
            receiver_id=f"dp{rank}",
            installed_version=str(2 - rank),
            state="POISONED",
            detail="engine-local ranks diverged after ModelExpress mutation",
        )
        for rank in range(2)
    ]

    class WriterLock:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

    class TokenizerHarness:
        def __init__(self, poison_error=None):
            self.model_update_lock = SimpleNamespace(writer_lock=WriterLock())
            self.requests = []
            self._modelexpress_poisoned = False
            self.poison_error = poison_error

        def auto_create_handle_loop(self):
            pass

        async def modelexpress_communicator(self, request):
            self.requests.append(request)
            if isinstance(request, MarkModelExpressPoisonedReqInput):
                if self.poison_error is not None:
                    raise self.poison_error
                return poisoned
            return [verified, failed]

        _merge_modelexpress_results = staticmethod(
            TokenizerControlMixin._merge_modelexpress_results
        )

        def _update_weight_version_if_provided(self, _version):
            raise AssertionError("divergent update must not advance the version")

    tokenizer = TokenizerHarness()
    result = asyncio.run(
        TokenizerControlMixin.update_weights_from_modelexpress(
            tokenizer, UpdateWeightsFromModelExpressReqInput(target_version="2")
        )
    )

    assert result.state == "POISONED"
    assert tokenizer._modelexpress_poisoned is True
    assert len(tokenizer.requests) == 2
    assert isinstance(tokenizer.requests[1], MarkModelExpressPoisonedReqInput)

    failed_fanout = TokenizerHarness(RuntimeError("poison fanout failed"))
    failed_result = asyncio.run(
        TokenizerControlMixin.update_weights_from_modelexpress(
            failed_fanout,
            UpdateWeightsFromModelExpressReqInput(target_version="2"),
        )
    )
    assert failed_fanout._modelexpress_poisoned is True
    assert failed_result.state == "POISONED"
    assert "reconciliation required" in failed_result.detail

    cancelled_fanout = TokenizerHarness(asyncio.CancelledError())
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(
            TokenizerControlMixin.update_weights_from_modelexpress(
                cancelled_fanout,
                UpdateWeightsFromModelExpressReqInput(target_version="2"),
            )
        )
    assert cancelled_fanout._modelexpress_poisoned is True
