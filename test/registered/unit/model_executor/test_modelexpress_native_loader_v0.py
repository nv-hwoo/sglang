# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

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
NativeInstallError = module.NativeInstallError
load_prepared_checkpoint = module.load_prepared_checkpoint


def runner_fixture():
    return SimpleNamespace(
        model=object(),
        model_config=SimpleNamespace(model_path="original-model", dtype=torch.float32),
        server_args=SimpleNamespace(
            model_path="original-model",
            load_format="auto",
            model_loader_extra_config={},
            download_dir=None,
        ),
        device="cpu",
    )


def test_native_loader_uses_prepared_path_without_reconfiguring_runner(tmp_path):
    runner = runner_fixture()
    loader = Mock()
    loader._get_weights_iterator.return_value = iter([("weight", torch.ones(1))])

    load_prepared_checkpoint(runner, tmp_path, loader_factory=lambda: loader)

    source = loader._get_weights_iterator.call_args.args[0]
    assert Path(source.model_or_path) == tmp_path
    loader.load_weights_and_postprocess.assert_called_once()
    assert runner.model_config.model_path == "original-model"
    assert runner.server_args.model_path == "original-model"
    assert runner.server_args.load_format == "auto"


def test_native_loader_classifies_setup_failure_before_write(tmp_path):
    with pytest.raises(NativeInstallError) as error:
        load_prepared_checkpoint(
            runner_fixture(),
            tmp_path,
            loader_factory=Mock(side_effect=RuntimeError("loader setup failed")),
        )

    assert error.value.mutation_started is False


def test_native_loader_classifies_load_failure_after_possible_write(tmp_path):
    loader = Mock()
    loader._get_weights_iterator.return_value = iter([("weight", torch.ones(1))])
    loader.load_weights_and_postprocess.side_effect = RuntimeError("load failed")

    with pytest.raises(NativeInstallError) as error:
        load_prepared_checkpoint(
            runner_fixture(), tmp_path, loader_factory=lambda: loader
        )

    assert error.value.mutation_started is True
