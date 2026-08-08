# SPDX-License-Identifier: Apache-2.0

"""SGLang adapter for the ModelExpress canonical receiver."""

from __future__ import annotations

import socket
from types import SimpleNamespace

import torch
from modelexpress.refit.receiver import (
    ModelExpressWeightReceiver,
    ReceiverConfig,
    ReceiverInstallError,
    build_weight_receiver,
)

NativeInstallError = ReceiverInstallError


def load_prepared_checkpoint(model_runner, path, *, loader_factory=None):
    injected_loader = loader_factory is not None
    try:
        if loader_factory is None:
            from sglang.srt.configs.load_config import LoadConfig, LoadFormat
            from sglang.srt.model_loader.loader import (
                DefaultModelLoader,
                get_model_loader,
            )

            loader = get_model_loader(
                LoadConfig(
                    load_format=LoadFormat.SAFETENSORS,
                    download_dir=model_runner.server_args.download_dir,
                    model_loader_extra_config=model_runner.server_args.model_loader_extra_config,
                ),
                model_runner.model_config,
            )
            if not isinstance(loader, DefaultModelLoader):
                raise TypeError("ModelExpress V0 requires DefaultModelLoader")
        else:
            loader = loader_factory()
        source = SimpleNamespace(
            model_or_path=str(path),
            revision=None,
            prefix="",
            fall_back_to_pt=False,
            model_config=model_runner.model_config,
        )
        weights = loader._get_weights_iterator(source)
    except Exception as error:
        raise ReceiverInstallError(str(error), mutation_started=False) from error

    try:
        if injected_loader:
            loader.load_weights_and_postprocess(
                model_runner.model, weights, torch.device(model_runner.device)
            )
        else:
            from sglang.srt.model_loader.utils import set_default_torch_dtype

            with set_default_torch_dtype(model_runner.model_config.dtype):
                loader.load_weights_and_postprocess(
                    model_runner.model, weights, torch.device(model_runner.device)
                )
        device = torch.get_device_module(model_runner.device)
        if hasattr(device, "synchronize"):
            device.synchronize()
    except Exception as error:
        raise ReceiverInstallError(str(error), mutation_started=True) from error


def build_modelexpress_weight_receiver(
    model_runner, *, catalog=None, s3_client=None, checkpoint=None
) -> ModelExpressWeightReceiver:
    args = model_runner.server_args
    if checkpoint is None:
        from sglang.srt.model_loader.loader import DefaultModelLoader

        if not isinstance(model_runner.loader, DefaultModelLoader):
            raise TypeError("ModelExpress V0 requires DefaultModelLoader")
        checkpoint, _files, _use_safetensors = model_runner.loader._prepare_weights(
            model_runner.model_config.model_path,
            model_runner.model_config.revision,
            False,
        )

    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    config = ReceiverConfig(
        model_id=args.modelexpress_model_id,
        catalog_endpoint=args.modelexpress_catalog_endpoint,
        initial_version=args.modelexpress_initial_version,
        preparation_cache_dir=args.modelexpress_preparation_cache_dir,
        ready_timeout_seconds=args.modelexpress_ready_timeout_seconds,
        s3_endpoint_url=args.modelexpress_delta_s3_endpoint,
    )
    return build_weight_receiver(
        config=config,
        receiver_id=f"{socket.gethostname()}:{rank}",
        launch_checkpoint=checkpoint,
        install_target=lambda target: load_prepared_checkpoint(
            model_runner, target["path"]
        ),
        catalog=catalog,
        s3_client=s3_client,
    )
