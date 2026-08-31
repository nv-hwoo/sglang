from __future__ import annotations

import hashlib
import logging
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import msgspec
import torch

from sglang.srt.constants import (
    GPU_MEMORY_ALL_TYPES,
    GPU_MEMORY_TYPE_CUDA_GRAPH,
    GPU_MEMORY_TYPE_KV_CACHE,
    GPU_MEMORY_TYPE_WEIGHTS,
)
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.io_struct import (
    ChecksumInfo,
    CheckWeightsReqInput,
    CheckWeightsReqOutput,
    DestroyWeightsUpdateGroupReqInput,
    DestroyWeightsUpdateGroupReqOutput,
    GetModelExpressStatusReqInput,
    GetWeightsByNameReqInput,
    GetWeightsByNameReqOutput,
    InitWeightsUpdateGroupReqInput,
    InitWeightsUpdateGroupReqOutput,
    MarkModelExpressPoisonedReqInput,
    ModelExpressWeightUpdateReqOutput,
    PrepareWeightsFromModelExpressReqInput,
    ReleaseMemoryOccupationReqInput,
    ReleaseMemoryOccupationReqOutput,
    ResumeMemoryOccupationReqInput,
    ResumeMemoryOccupationReqOutput,
    UpdateWeightFromDiskReqInput,
    UpdateWeightFromDiskReqOutput,
    UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromDistributedReqOutput,
    UpdateWeightsFromIPCReqInput,
    UpdateWeightsFromIPCReqOutput,
    UpdateWeightsFromModelExpressReqInput,
    UpdateWeightsFromTensorReqInput,
    UpdateWeightsFromTensorReqOutput,
)
from sglang.srt.runtime_context import get_model

logger = logging.getLogger(__name__)


def _get_draft_model_runner(draft_worker):
    # DFlash / FrozenKVMTP workers expose draft_model_runner directly
    runner = getattr(draft_worker, "draft_model_runner", None)
    if runner is not None:
        return runner
    # EAGLEWorkerV2: _draft_worker.draft_runner
    inner = getattr(draft_worker, "_draft_worker", None)
    if inner is not None:
        runner = getattr(inner, "draft_runner", None)
        if runner is not None:
            return runner
    return None


def _merge_checksum_payloads(target: Dict, draft: Dict) -> Dict:
    merged_checksums = dict(target["checksums"])
    for name, chk in draft["checksums"].items():
        merged_checksums[f"draft.{name}"] = chk
    h = hashlib.sha256()
    for name in sorted(merged_checksums):
        h.update(name.encode())
        h.update(merged_checksums[name].encode())
    target["checksums"] = merged_checksums
    target["per_gpu_checksum"] = h.hexdigest()
    return target


@dataclass(kw_only=True, slots=True)
class SchedulerWeightUpdaterManager:
    tp_worker: Any
    draft_worker: Any
    tp_cpu_group: Any
    memory_saver_adapter: Any
    flush_cache: Callable[..., bool]
    is_fully_idle: Callable[..., bool]
    scheduler: Optional[Any] = None
    metrics_collector: Optional[Any] = None
    modelexpress_generator: Any = field(default=None, init=False)
    modelexpress_staged: Any = field(default=None, init=False)
    modelexpress_installed_version: Optional[str] = field(default=None, init=False)
    modelexpress_state: Optional[str] = field(default=None, init=False)
    modelexpress_detail: str = field(default="", init=False)
    offload_tags: set = field(default_factory=set)
    stashed_model_static_state: Any = None

    @contextmanager
    def _observe_weight_load(self, source: str) -> Iterator[None]:
        # Edge-trigger weight_load_duration_seconds at the end of each
        # update_weights_from_* call. Engine is paused during the update so
        # the periodic log_stats path can't carry this.
        # `source` distinguishes disk vs distributed vs tensor vs ipc.
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if self.metrics_collector is not None:
                self.metrics_collector.observe_weight_load(
                    time.perf_counter() - t0, source
                )

    def _observe_modelexpress_metrics(self, metrics: Dict[str, float]) -> None:
        if self.metrics_collector is None:
            return
        try:
            self.metrics_collector.observe_modelexpress_metrics(metrics)
        except Exception:
            logger.warning("Failed to record ModelExpress metrics", exc_info=True)

    def flush_cache_after_weight_update(self, recv_req) -> None:
        if recv_req.flush_cache:
            flush_cache_success = self.flush_cache(
                empty_cache=recv_req.torch_empty_cache
            )
            assert flush_cache_success, "Cache flush failed after updating weights"

    def record_weight_version_after_update(self, weight_version: Optional[str]) -> None:
        self.scheduler.record_weight_version_change(new_version=weight_version)

    def update_weights_from_disk(self, recv_req: UpdateWeightFromDiskReqInput):
        """In-place update of the weights from disk."""
        with self._observe_weight_load("disk"):
            success, message = self.tp_worker.update_weights_from_disk(recv_req)
            tp_success = success
            if success and self.draft_worker is not None:
                success, message = self.draft_worker.update_weights_from_disk(recv_req)
            if tp_success:
                self.flush_cache_after_weight_update(recv_req)
            if success:
                self.record_weight_version_after_update(recv_req.weight_version)
            else:
                logger.error(message)
            return UpdateWeightFromDiskReqOutput(
                success=success, message=message, num_paused_requests=0
            )

    def _modelexpress_output(
        self,
        *,
        success: bool = True,
        detail: Optional[str] = None,
        metrics: Optional[Dict[str, float]] = None,
    ) -> ModelExpressWeightUpdateReqOutput:
        generator = self._modelexpress_generator()
        return ModelExpressWeightUpdateReqOutput(
            success=success,
            receiver_id=generator.worker_id,
            installed_version=self.modelexpress_installed_version,
            state=self.modelexpress_state,
            detail=self.modelexpress_detail if detail is None else detail,
            metrics=metrics or {},
        )

    def _modelexpress_generator(self):
        if self.draft_worker is not None:
            raise RuntimeError("ModelExpress does not support draft models")
        if self.modelexpress_generator is None:
            from modelexpress_rl import (
                ModelExpressGeneratorClient,
                ModelExpressGeneratorConfig,
                ObjectStorageGeneratorConfig,
                ObjectStorageType,
                SglangGeneratorContext,
            )

            model_runner = self.tp_worker.model_runner
            args = model_runner.server_args
            checkpoint, _, _ = model_runner.loader._prepare_weights(
                model_runner.model_config.model_path,
                model_runner.model_config.revision,
                False,
            )
            self.modelexpress_generator = ModelExpressGeneratorClient.initialize(
                ModelExpressGeneratorConfig(
                    engine_context=SglangGeneratorContext(model_runner),
                    model_name=args.modelexpress_model_name,
                    server_url=args.modelexpress_server_url,
                    object_storage=ObjectStorageGeneratorConfig(
                        storage_type=ObjectStorageType.S3,
                        initial_base_version_id=(
                            args.modelexpress_initial_base_version_id
                        ),
                        seed_checkpoint_path=checkpoint,
                        refit_checkpoint_dir=(args.modelexpress_preparation_cache_dir),
                        endpoint_url=args.modelexpress_s3_endpoint_url,
                    ),
                )
            )
            self.modelexpress_installed_version = (
                args.modelexpress_initial_base_version_id
            )
            self.modelexpress_state = "VERIFIED"
        return self.modelexpress_generator

    def _modelexpress_tp_agreement(
        self,
        output: ModelExpressWeightUpdateReqOutput,
        mutation_phase: bool,
    ) -> ModelExpressWeightUpdateReqOutput:
        if not torch.distributed.is_initialized():
            return output
        world_size = torch.distributed.get_world_size(group=self.tp_cpu_group)
        if world_size <= 1:
            return output
        results: List[ModelExpressWeightUpdateReqOutput] = [output] * world_size
        torch.distributed.all_gather_object(results, output, group=self.tp_cpu_group)
        identities = {
            (item.success, item.installed_version, item.state) for item in results
        }
        if len(identities) == 1:
            output.metrics = {
                key: max(item.metrics.get(key, 0.0) for item in results)
                for key in {key for item in results for key in item.metrics}
            }
            return output
        detail = "engine-local TP ranks disagree on ModelExpress target identity"
        if mutation_phase and any(
            item.success or item.state == "POISONED" for item in results
        ):
            self.modelexpress_state = "POISONED"
            self.modelexpress_detail = detail
            return self._modelexpress_output(success=False)
        failures = [item for item in results if not item.success]
        if failures:
            return failures[0]
        return ModelExpressWeightUpdateReqOutput(
            success=False,
            receiver_id=output.receiver_id,
            installed_version=output.installed_version,
            state=output.state,
            detail=detail,
        )

    def prepare_weights_from_modelexpress(
        self, recv_req: PrepareWeightsFromModelExpressReqInput
    ):
        from modelexpress_rl import WeightVersionRef

        generator = self._modelexpress_generator()
        if self.modelexpress_state == "POISONED":
            return self._modelexpress_tp_agreement(
                self._modelexpress_output(success=False),
                mutation_phase=False,
            )
        try:
            if self.modelexpress_staged is None:
                self.modelexpress_staged = generator.stage_weight(
                    version=WeightVersionRef(recv_req.target_version)
                )
            elif self.modelexpress_staged.version_id != recv_req.target_version:
                raise RuntimeError("another ModelExpress target is already prepared")
            self.modelexpress_state = "VERIFIED"
            self.modelexpress_detail = ""
            metrics = self.modelexpress_staged.metrics
            self._observe_modelexpress_metrics(metrics)
            output = self._modelexpress_output(metrics=metrics)
        except Exception as exc:
            self.modelexpress_state = "FAILED"
            self.modelexpress_detail = str(exc)
            output = self._modelexpress_output(success=False)
        return self._modelexpress_tp_agreement(output, mutation_phase=False)

    def update_weights_from_modelexpress(
        self, recv_req: UpdateWeightsFromModelExpressReqInput
    ):
        generator = self._modelexpress_generator()
        staged = self.modelexpress_staged
        if staged is None or staged.version_id != recv_req.target_version:
            precheck = self._modelexpress_output(
                success=False,
                detail="requested target is not prepared",
            )
        else:
            precheck = self._modelexpress_output()
        precheck = self._modelexpress_tp_agreement(precheck, mutation_phase=False)
        if not precheck.success:
            return precheck

        metrics = {}
        success = False
        try:
            metrics = generator.apply_weight(staged)
            self._observe_modelexpress_metrics(metrics)
            success = True
        except Exception as exc:
            if staged.applied:
                self.modelexpress_installed_version = staged.version_id
                self.modelexpress_state = "VERIFIED"
                success = True
            else:
                self.modelexpress_state = "FAILED"
            self.modelexpress_detail = str(exc)
        finally:
            try:
                staged.release()
            except Exception as exc:
                suffix = f"staged cleanup failed: {exc}"
                self.modelexpress_detail = (
                    f"{self.modelexpress_detail}; {suffix}"
                    if self.modelexpress_detail
                    else suffix
                )
            finally:
                self.modelexpress_staged = None

        if success:
            self.modelexpress_installed_version = staged.version_id
            self.modelexpress_state = "VERIFIED"
            try:
                cache_valid = self.flush_cache(empty_cache=False)
            except Exception as exc:
                self.modelexpress_state = "POISONED"
                self.modelexpress_detail = (
                    f"cache invalidation raised after install: {exc}"
                )
                success = False
            else:
                if not cache_valid:
                    self.modelexpress_state = "POISONED"
                    self.modelexpress_detail = "cache invalidation failed after install"
                    success = False
        return self._modelexpress_tp_agreement(
            self._modelexpress_output(success=success, metrics=metrics),
            mutation_phase=True,
        )

    def mark_modelexpress_poisoned(self, recv_req: MarkModelExpressPoisonedReqInput):
        self.modelexpress_state = "POISONED"
        self.modelexpress_detail = recv_req.detail
        return self._modelexpress_tp_agreement(
            self._modelexpress_output(success=False), mutation_phase=True
        )

    def get_modelexpress_status(self, _recv_req: GetModelExpressStatusReqInput):
        output = self._modelexpress_output()
        return self._modelexpress_tp_agreement(output, mutation_phase=False)

    def init_weights_update_group(self, recv_req: InitWeightsUpdateGroupReqInput):
        """Initialize the online model parameter update group."""
        success, message = self.tp_worker.init_weights_update_group(recv_req)
        return InitWeightsUpdateGroupReqOutput(success=success, message=message)

    def destroy_weights_update_group(
        self,
        recv_req: DestroyWeightsUpdateGroupReqInput,
    ):
        """Destroy the online model parameter update group."""
        success, message = self.tp_worker.destroy_weights_update_group(recv_req)
        return DestroyWeightsUpdateGroupReqOutput(success=success, message=message)

    def update_weights_from_distributed(
        self,
        recv_req: UpdateWeightsFromDistributedReqInput,
    ) -> Tuple[bool, str]:
        """Update the online model parameter."""
        with self._observe_weight_load("distributed"):
            success, message = self.tp_worker.update_weights_from_distributed(recv_req)
            if success:
                self.flush_cache_after_weight_update(recv_req)
                self.record_weight_version_after_update(recv_req.weight_version)
            else:
                logger.error(message)
            return UpdateWeightsFromDistributedReqOutput(
                success=success, message=message
            )

    def update_weights_from_tensor(self, recv_req: UpdateWeightsFromTensorReqInput):
        """Update the online model parameter from tensors."""
        with self._observe_weight_load("tensor"):
            if recv_req.disable_draft_model:
                worker = self.tp_worker
            else:
                worker = self.draft_worker or self.tp_worker
            success, message = worker.update_weights_from_tensor(recv_req)
            if success:
                self.flush_cache_after_weight_update(recv_req)
                self.record_weight_version_after_update(recv_req.weight_version)
            else:
                logger.error(message)
            torch.distributed.barrier(group=self.tp_cpu_group)
            return UpdateWeightsFromTensorReqOutput(success=success, message=message)

    def update_weights_from_ipc(self, recv_req: UpdateWeightsFromIPCReqInput):
        """Update the online model parameter from IPC for checkpoint-engine integration."""
        with self._observe_weight_load("ipc"):
            success, message = self.tp_worker.update_weights_from_ipc(recv_req)
            tp_success = success
            if success and self.draft_worker is not None:
                success, message = self.draft_worker.update_weights_from_ipc(recv_req)
            if tp_success:
                self.flush_cache_after_weight_update(recv_req)
            if success:
                self.record_weight_version_after_update(recv_req.weight_version)
            else:
                logger.error(message)
            torch.distributed.barrier(group=self.tp_cpu_group)
            return UpdateWeightsFromIPCReqOutput(success=success, message=message)

    def get_weights_by_name(self, recv_req: GetWeightsByNameReqInput):
        parameter = self.tp_worker.get_weights_by_name(recv_req)
        return GetWeightsByNameReqOutput(parameter=parameter)

    def _assert_weight_cache_inactive(self, op: str) -> None:
        """Reject freeing/restoring model weights while the CUDA IPC weight
        cache is active: the weights are shared with the daemon via CUDA IPC, so
        freeing them would leave the daemon and every peer pointing at released
        memory.
        """
        mode = get_model().weight_cache_mode
        if mode != "off":
            raise RuntimeError(
                f"[weight_cache] {op} of model weights is not supported while the "
                f"weight cache is active (--weight-cache-mode {mode}): the weights "
                f"are shared with the daemon via CUDA IPC, so freeing them would "
                f"corrupt the daemon's master copy and every co-attached engine. "
                f"Restart with --weight-cache-mode off to use this operation."
            )

    def release_memory_occupation(self, recv_req: ReleaseMemoryOccupationReqInput):
        assert self.is_fully_idle(), (
            "release_memory_occupation should be called only when server is idle."
        )

        tags = recv_req.tags

        if tags is None or len(tags) == 0:
            tags = GPU_MEMORY_ALL_TYPES

        for tag in tags:
            self.offload_tags.add(tag)

        if GPU_MEMORY_TYPE_KV_CACHE in tags:
            scheduler = self.scheduler
            if scheduler is not None:
                if scheduler.disaggregation_mode == DisaggregationMode.DECODE:
                    for queue_name in (
                        "disagg_decode_transfer_queue",
                        "disagg_decode_prealloc_queue",
                    ):
                        queue = getattr(scheduler, queue_name, None)
                        if queue is not None:
                            queue.release_memory_occupation()
                elif scheduler.disaggregation_mode == DisaggregationMode.PREFILL:
                    queue = getattr(scheduler, "disagg_prefill_bootstrap_queue", None)
                    if queue is not None:
                        queue.release_memory_occupation()
            self.memory_saver_adapter.pause(GPU_MEMORY_TYPE_KV_CACHE)
            self.flush_cache()

        if GPU_MEMORY_TYPE_WEIGHTS in tags:
            self._assert_weight_cache_inactive("release_memory_occupation")
            self.stashed_model_static_state = _export_static_state(
                self.tp_worker.model_runner.model
            )
            torch.distributed.barrier(self.tp_cpu_group)
            self.memory_saver_adapter.pause(GPU_MEMORY_TYPE_WEIGHTS)

        if GPU_MEMORY_TYPE_CUDA_GRAPH in tags:
            self.memory_saver_adapter.pause(GPU_MEMORY_TYPE_CUDA_GRAPH)

        torch.get_device_module().synchronize()

        return ReleaseMemoryOccupationReqOutput()

    def resume_memory_occupation(self, recv_req: ResumeMemoryOccupationReqInput):
        tags = recv_req.tags

        if tags is None or len(tags) == 0:
            tags = GPU_MEMORY_ALL_TYPES

        for tag in tags:
            self.offload_tags.remove(tag)

        if GPU_MEMORY_TYPE_CUDA_GRAPH in tags:
            self.memory_saver_adapter.resume(GPU_MEMORY_TYPE_CUDA_GRAPH)

        if GPU_MEMORY_TYPE_WEIGHTS in tags:
            self._assert_weight_cache_inactive("resume_memory_occupation")
            self.memory_saver_adapter.resume(GPU_MEMORY_TYPE_WEIGHTS)
            torch.distributed.barrier(self.tp_cpu_group)
            _import_static_state(
                self.tp_worker.model_runner.model,
                self.stashed_model_static_state,
            )
            del self.stashed_model_static_state

        if GPU_MEMORY_TYPE_KV_CACHE in tags:
            self.memory_saver_adapter.resume(GPU_MEMORY_TYPE_KV_CACHE)
            scheduler = self.scheduler
            if scheduler is not None:
                if scheduler.disaggregation_mode == DisaggregationMode.DECODE:
                    for queue_name in (
                        "disagg_decode_transfer_queue",
                        "disagg_decode_prealloc_queue",
                    ):
                        queue = getattr(scheduler, queue_name, None)
                        if queue is not None:
                            queue.resume_memory_occupation()
                elif scheduler.disaggregation_mode == DisaggregationMode.PREFILL:
                    queue = getattr(scheduler, "disagg_prefill_bootstrap_queue", None)
                    if queue is not None:
                        queue.resume_memory_occupation()

        return ResumeMemoryOccupationReqOutput()

    def check_weights(self, recv_req: CheckWeightsReqInput):
        try:
            payload = self.tp_worker.model_runner.check_weights(
                action=recv_req.action, allow_quant_error=recv_req.allow_quant_error
            )

            if self.draft_worker is not None:
                draft_runner = _get_draft_model_runner(self.draft_worker)
                if draft_runner is not None:
                    draft_payload = draft_runner.check_weights(
                        action=recv_req.action,
                        allow_quant_error=recv_req.allow_quant_error,
                    )
                    if payload is not None and draft_payload is not None:
                        payload = _merge_checksum_payloads(payload, draft_payload)

            tp_size = torch.distributed.get_world_size(group=self.tp_cpu_group)
            if tp_size > 1 and payload is not None:
                all_payloads = [None] * tp_size
                torch.distributed.all_gather_object(
                    all_payloads, payload, group=self.tp_cpu_group
                )
                payload = all_payloads
            if payload is not None:
                # Normalize to one ChecksumInfo per rank so the wire shape is a
                # uniform List[ChecksumInfo] (tp==1 becomes a single-element list).
                per_rank = payload if isinstance(payload, list) else [payload]
                payload = [msgspec.convert(p, ChecksumInfo) for p in per_rank]
            return CheckWeightsReqOutput(
                success=True, message="Success.", payload=payload
            )
        except Exception as e:
            logger.warning(f"check_weights see error: {e}")
            traceback.print_exc()
            return CheckWeightsReqOutput(success=False, message=f"{e}")

    def save_remote_model(self, params):
        url = params["url"]

        self.tp_worker.model_runner.weight_exporter.save_remote_model(url)

        if self.draft_worker is not None:
            draft_url = params.get("draft_url", None)
            assert draft_url is not None, (
                "draft_url must be provided when draft model is enabled"
            )
            self.draft_worker.model_runner.weight_exporter.save_remote_model(draft_url)

    def save_sharded_model(self, params):
        self.tp_worker.model_runner.weight_exporter.save_sharded_model(
            path=params["path"],
            pattern=params["pattern"],
            max_size=params["max_size"],
        )


def _export_static_state(model):
    return dict(
        buffers=[
            (name, buffer.detach().clone()) for name, buffer in model.named_buffers()
        ]
    )


def _import_static_state(model, static_params):
    with torch.inference_mode():
        self_named_buffers = dict(model.named_buffers())
        for name, tensor in static_params["buffers"]:
            self_named_buffers[name][...] = tensor
