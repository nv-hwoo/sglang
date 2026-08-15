# SPDX-License-Identifier: Apache-2.0

import asyncio
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

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
from sglang.srt.managers.scheduler_components import (  # noqa: E402
    weight_updater as weight_updater_module,
)
from sglang.srt.managers.scheduler_components.weight_updater import (  # noqa: E402
    SchedulerWeightUpdaterManager,
)
from sglang.srt.managers.tokenizer_control_mixin import (  # noqa: E402
    TokenizerControlMixin,
)
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


class Receiver:
    def __init__(self):
        self.initialized = 0
        self.prepared = []
        self.metrics = [
            {"perf/mx_receive_prepare_time": 2.0},
            {"perf/mx_receive_install_time": 3.0},
        ]

    def initialize(self):
        self.initialized += 1

    def start_weight_update(self, version):
        self.prepared.append(version)

    def pop_metrics(self):
        return self.metrics.pop(0)

    @property
    def prepared_identity(self):
        # Mirrors ModelExpressWeightReceiver: None until a target is prepared.
        if not self.prepared:
            return None
        return SimpleNamespace(
            target_version=self.prepared[-1], target_digest="sha256:target"
        )

    def update_weights(self):
        return SimpleNamespace(
            success=True,
            receiver_id="tp0",
            installed_version="2",
            state=SimpleNamespace(name="VERIFIED"),
            target_digest="sha256:target",
            detail="",
        )

    def status(self):
        return SimpleNamespace(
            receiver_id="tp0",
            installed_version="2",
            state=SimpleNamespace(name="VERIFIED"),
            detail="",
        )

    def mark_poisoned(self, detail):
        return SimpleNamespace(
            success=False,
            receiver_id="tp0",
            installed_version="2",
            state=SimpleNamespace(name="POISONED"),
            target_digest=None,
            detail=detail,
        )


def manager(receiver):
    value = object.__new__(SchedulerWeightUpdaterManager)
    value.tp_worker = SimpleNamespace(
        model_runner=SimpleNamespace(
            server_args=SimpleNamespace(
                modelexpress_model_id="model",
                modelexpress_catalog_endpoint="mx:8001",
                modelexpress_initial_version="0",
                modelexpress_preparation_cache_dir="/tmp/mx-cache",
                modelexpress_ready_timeout_seconds=123,
                modelexpress_delta_s3_endpoint="http://minio:9000",
            )
        )
    )
    value.modelexpress_receiver = receiver
    value.draft_worker = None
    value.tp_cpu_group = object()
    value.flush_cache = lambda **_kwargs: True
    return value


def test_receiver_is_built_once_on_first_manager_access(monkeypatch):
    receiver = Receiver()
    updater = manager(None)
    built = []

    def build(backend, **kwargs):
        built.append((backend, kwargs))
        return receiver

    class ReceiverConfig(SimpleNamespace):
        def __init__(self, **values):
            super().__init__(**values)

    class RolloutBackend:
        SGLANG = "sglang"

    modelexpress = ModuleType("modelexpress")
    modelexpress.__path__ = []
    refit = ModuleType("modelexpress.refit")
    refit.__path__ = []
    factory_module = ModuleType("modelexpress.refit.factory")
    factory_module.RolloutBackend = RolloutBackend
    factory_module.build_delta_receiver = build
    receiver_module = ModuleType("modelexpress.refit.receiver")
    receiver_module.ReceiverConfig = ReceiverConfig
    monkeypatch.setitem(sys.modules, "modelexpress", modelexpress)
    monkeypatch.setitem(sys.modules, "modelexpress.refit", refit)
    monkeypatch.setitem(sys.modules, "modelexpress.refit.factory", factory_module)
    monkeypatch.setitem(sys.modules, "modelexpress.refit.receiver", receiver_module)
    monkeypatch.setattr(
        weight_updater_module,
        "socket",
        SimpleNamespace(gethostname=lambda: "host"),
        raising=False,
    )

    status = updater.get_modelexpress_status(GetModelExpressStatusReqInput())
    prepared = updater.prepare_weights_from_modelexpress(
        PrepareWeightsFromModelExpressReqInput(target_version="2")
    )

    assert status.state == "VERIFIED"
    assert prepared.success is True
    assert len(built) == 1
    backend, kwargs = built[0]
    assert backend == RolloutBackend.SGLANG
    assert kwargs["model_runner"] is updater.tp_worker.model_runner
    assert kwargs["receiver_id"] == "host:0"
    assert vars(kwargs["config"]) == {
        "model_id": "model",
        "catalog_endpoint": "mx:8001",
        "initial_version": "0",
        "preparation_cache_dir": "/tmp/mx-cache",
        "ready_timeout_seconds": 123,
        "s3_endpoint_url": "http://minio:9000",
    }
    assert receiver.initialized == 1
    assert updater.modelexpress_receiver is receiver


def test_scheduler_handlers_are_thin_receiver_forwarders():
    receiver = Receiver()
    updater = manager(receiver)

    prepared = updater.prepare_weights_from_modelexpress(
        PrepareWeightsFromModelExpressReqInput(target_version="2")
    )
    installed = updater.update_weights_from_modelexpress(
        UpdateWeightsFromModelExpressReqInput(target_version="2")
    )
    status = updater.get_modelexpress_status(GetModelExpressStatusReqInput())

    assert receiver.prepared == ["2"]
    assert prepared.success is True
    assert prepared.metrics == {"perf/mx_receive_prepare_time": 2.0}
    assert installed.success is True
    assert installed.installed_version == "2"
    assert installed.target_digest == "sha256:target"
    assert installed.metrics == {"perf/mx_receive_install_time": 3.0}
    assert status.state == "VERIFIED"


def test_receive_metrics_merge_by_max_latency():
    first = ModelExpressWeightUpdateReqOutput(
        success=True,
        receiver_id="dp0",
        installed_version="2",
        state="VERIFIED",
        target_digest="sha256:target",
        metrics={"perf/mx_receive_prepare_time": 2.0},
    )
    second = ModelExpressWeightUpdateReqOutput(
        success=True,
        receiver_id="dp1",
        installed_version="2",
        state="VERIFIED",
        target_digest="sha256:target",
        metrics={"perf/mx_receive_prepare_time": 3.0},
    )

    result = TokenizerControlMixin._merge_modelexpress_results(
        [first, second], mutation_phase=False
    )

    assert result.metrics == {"perf/mx_receive_prepare_time": 3.0}


def test_installing_an_unprepared_target_is_refused_without_mutating():
    receiver = Receiver()
    updater = manager(receiver)

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
    receiver = Receiver()
    updater = manager(receiver)
    receiver.start_weight_update("2")

    def fail_flush(**_kwargs):
        raise RuntimeError("cache flush exploded")

    updater.flush_cache = fail_flush

    result = updater.update_weights_from_modelexpress(
        UpdateWeightsFromModelExpressReqInput(target_version="2")
    )

    assert result.success is False
    assert result.state == "POISONED"
    assert "cache invalidation raised after install" in result.detail


def test_tp_disagreement_after_mutation_poisoned_the_whole_engine(monkeypatch):
    receiver = Receiver()
    updater = manager(receiver)
    local = ModelExpressWeightUpdateReqOutput(
        success=True,
        receiver_id="tp0",
        installed_version="2",
        state="VERIFIED",
        target_digest="sha256:target",
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
        target_digest="sha256:target",
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
        target_digest="sha256:target",
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
