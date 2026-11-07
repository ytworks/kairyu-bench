"""Formal vLLM general-plugin/custom-loader entrypoint; never a live hotpatch."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from vllm.model_executor.model_loader.default_loader import DefaultModelLoader

from .core import FORMAT, VLLM_FILES, Overlay, canonical, file_sha


class SparseOverlayLoader(DefaultModelLoader):
    def __init__(self, load_config: Any):
        extra = load_config.model_loader_extra_config
        if not isinstance(extra, dict) or set(extra) != {"overlay_plan", "overlay_plan_sha256"}:
            raise ValueError("overlay requires only an exact plan path and SHA")
        self.overlay = Overlay({"path": extra["overlay_plan"], "sha256": extra["overlay_plan_sha256"]})
        import vllm
        root = Path(vllm.__file__).parent
        self.vllm_root = root
        actual = {name: file_sha(root / name) for name in VLLM_FILES}
        if self.overlay.plan["vllm_source_sha256"] != actual:
            raise ValueError("installed vLLM mapping/loader source differs from reviewed pins")
        # DefaultModelLoader rejects unknown formats and extra keys internally.
        # Use an independent config, retaining the outer registered format unmodified.
        super().__init__(replace(load_config, load_format="safetensors",
                                 model_loader_extra_config={}, safetensors_load_strategy="lazy"))
        self.invoked = False

    def get_all_weights(self, model_config: Any, model: Any):
        if getattr(model, "secondary_weights", ()):
            raise ValueError("secondary checkpoint sources are not supported")
        yield from self.overlay.iterator(super().get_all_weights(model_config, model), model_config.model)

    def load_model(self, vllm_config: Any, model_config: Any, prefix: str = ""):
        import torch
        import torch.distributed as dist
        from vllm.distributed.parallel_state import get_pp_group, get_tp_group
        if self.invoked:
            raise ValueError("loader instance cannot be reused")
        self.invoked = True
        role = self.overlay.role(model_config.model)
        if role == "target" and model_config.hf_config.architectures != ["Gemma4ForConditionalGeneration"]:
            raise ValueError("only the reviewed Gemma4 multimodal target mapping is supported")
        parallel = vllm_config.parallel_config
        if (model_config.dtype != torch.bfloat16 or model_config.quantization is not None
                or get_pp_group().world_size != 1 or parallel.data_parallel_size != 1
                or parallel.enable_expert_parallel or vllm_config.lora_config is not None):
            raise ValueError("only dense BF16 PP1/DP1 without LoRA/EP is supported")
        group = get_tp_group()
        if group.world_size != self.overlay.tp_size:
            raise ValueError("TP size differs from overlay proof contract")
        path = self.overlay.receipt_dir / f"{role}-rank-{group.rank_in_group}.json"
        if path.exists():
            raise ValueError("rank receipt already exists; use a fresh owner invocation")
        model = super().load_model(vllm_config, model_config, prefix)
        if {name: file_sha(self.vllm_root / name) for name in VLLM_FILES} != self.overlay.plan["vllm_source_sha256"]:
            raise ValueError("vLLM source changed during load")
        self.overlay.check()
        if role == "draft_bypass":
            proof = {"schema_version": 1, "status": "pinned_draft_bypassed",
                     "rank": group.rank_in_group, "draft_tree_sha256": self.overlay.draft.tree,
                     "logical_overlay_sha256": self.overlay.logical_id,
                     "plan": self.overlay.plan_record,
                     "source_proof": "prior_full_file_sha256_plus_exact_current_stat",
                     "draft_tensor_edits": 0}
        else:
            try:
                local = self.overlay.rank_proof(model, group.rank_in_group)
            except Exception as error:
                # Reach the same CPU collective on every rank even if local readback fails.
                local = {"rank": group.rank_in_group, "error": type(error).__name__}
            proofs: list[Any] = [None] * group.world_size
            if group.world_size == 1:
                proofs[0] = local
            else:
                dist.all_gather_object(proofs, local, group=group.cpu_group)
            proof = self.overlay.aggregate(proofs)
        # The directory belongs to a fresh owner invocation. No overwrite or stale reuse.
        with path.open("xb") as stream:
            stream.write(canonical(proof) + b"\n")
        return model


def register() -> None:
    from vllm.model_executor.model_loader import register_model_loader
    register_model_loader(FORMAT)(SparseOverlayLoader)
