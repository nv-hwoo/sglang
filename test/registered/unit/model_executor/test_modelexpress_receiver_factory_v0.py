# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

MODULE = (
    Path(__file__).parents[4]
    / "python/sglang/srt/model_executor/model_runner_components/modelexpress_weight_receiver.py"
)
spec = importlib.util.spec_from_file_location("modelexpress_weight_receiver", MODULE)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


def runner(tmp_path):
    return SimpleNamespace(
        model=object(),
        device="cpu",
        model_config=SimpleNamespace(
            model_path=str(tmp_path / "hf"), revision=None, dtype=torch.float32
        ),
        server_args=SimpleNamespace(
            modelexpress_model_id="model",
            modelexpress_catalog_endpoint="mx:8001",
            modelexpress_delta_s3_endpoint="http://minio:9000",
            modelexpress_preparation_cache_dir=str(tmp_path / "cache"),
            modelexpress_initial_version="0",
            modelexpress_ready_timeout_seconds=123,
            download_dir=None,
            model_loader_extra_config=None,
        ),
    )


def test_factory_is_thin_config_and_native_install_adapter(tmp_path, monkeypatch):
    captured = {}
    installed = []
    sentinel = object()

    def build(**kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(module, "build_weight_receiver", build)
    monkeypatch.setattr(module.socket, "gethostname", lambda: "host")
    monkeypatch.setattr(
        module,
        "load_prepared_checkpoint",
        lambda model_runner, path: installed.append((model_runner, path)),
    )
    model_runner = runner(tmp_path)
    checkpoint = tmp_path / "hf"

    result = module.build_modelexpress_weight_receiver(
        model_runner,
        checkpoint=checkpoint,
        catalog="catalog",
        s3_client="s3",
    )

    assert result is sentinel
    assert captured["config"].model_id == "model"
    assert captured["config"].catalog_endpoint == "mx:8001"
    assert captured["config"].s3_endpoint_url == "http://minio:9000"
    assert captured["config"].initial_version == "0"
    assert captured["config"].preparation_cache_dir == str(tmp_path / "cache")
    assert captured["config"].ready_timeout_seconds == 123
    assert captured["receiver_id"] == "host:0"
    assert captured["launch_checkpoint"] == checkpoint
    assert captured["catalog"] == "catalog"
    assert captured["s3_client"] == "s3"

    target = {"path": tmp_path / "prepared"}
    captured["install_target"](target)
    assert installed == [(model_runner, target["path"])]


def test_native_loader_setup_failure_is_pre_mutation(tmp_path):
    class Loader:
        def _get_weights_iterator(self, _source):
            raise RuntimeError("setup")

    with pytest.raises(module.NativeInstallError) as captured:
        module.load_prepared_checkpoint(
            runner(tmp_path), tmp_path / "prepared", loader_factory=Loader
        )

    assert not captured.value.mutation_started


def test_native_loader_failure_after_weights_iterator_is_post_mutation(tmp_path):
    class Loader:
        def _get_weights_iterator(self, _source):
            return iter(())

        def load_weights_and_postprocess(self, *_args):
            raise RuntimeError("load")

    with pytest.raises(module.NativeInstallError) as captured:
        module.load_prepared_checkpoint(
            runner(tmp_path), tmp_path / "prepared", loader_factory=Loader
        )

    assert captured.value.mutation_started
