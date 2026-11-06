"""Exact, sparse BF16 input overlay and post-TP proof; no model construction."""
from __future__ import annotations

import hashlib
import json
import re
import struct
from pathlib import Path
from typing import Any, Iterable, Iterator

from kairyu_bench.wsp.weights import checkpoint as reader
from kairyu_bench.wsp.weights import editor

FORMAT = "kairyu_sparse_bf16_v1"
PATTERN = re.compile(
    r"model\.language_model\.layers\.(\d+)\.(self_attn\.o_proj|mlp\.(?:gate|up|down)_proj)\.weight"
)
# This mapping is reviewed against these installed source files, not version text alone.
VLLM_FILES = (
    "model_executor/model_loader/__init__.py",
    "model_executor/model_loader/base_loader.py",
    "model_executor/model_loader/default_loader.py",
    "model_executor/model_loader/weight_utils.py",
    "model_executor/model_loader/utils.py",
    "model_executor/models/gemma4.py",
    "model_executor/models/gemma4_mm.py",
    "model_executor/layers/linear.py",
    "distributed/parallel_state.py",
)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def file_sha(path: Path) -> str:
    return digest(path.read_bytes())


def stamp(path: Path) -> tuple[int, int, int, int, int]:
    s = path.stat()
    return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns


def exact_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or path != path.resolve() or path.is_symlink():
        raise ValueError("absolute canonical paths without symlinks required")
    return path


def implementation_identity() -> dict[str, str]:
    root = Path(__file__).parent
    return {
        **{name: file_sha(root / name) for name in ("core.py", "loader.py", "__init__.py")},
        "checkpoint.py": file_sha(Path(reader.__file__)),
        "editor.py": file_sha(Path(editor.__file__)),
    }


class Inputs:
    def __init__(self) -> None:
        self.pins: dict[Path, tuple[str, tuple[int, int, int, int, int]]] = {}

    def read(self, record: dict[str, str]) -> dict[str, Any]:
        path = exact_path(record["path"])
        before = stamp(path)
        raw = path.read_bytes()
        pin = (digest(raw), before)
        if pin[0] != record["sha256"] or stamp(path) != before:
            raise ValueError("input SHA/stat mismatch")
        if path in self.pins and self.pins[path] != pin:
            raise ValueError("conflicting input pins")
        self.pins[path] = pin
        return reader._json(raw)

    def check(self) -> None:
        for path, (sha, old) in self.pins.items():
            if stamp(path) != old or file_sha(path) != sha:
                raise ValueError("overlay input changed")


class Source:
    """Prior full-file hash plus exact current stat; never a new bulk hash."""
    def __init__(self, declaration: dict[str, Any], inputs: Inputs):
        self.path = exact_path(declaration["path"])
        if not self.path.is_dir():
            raise ValueError("source must be a local checkpoint directory")
        self.proof = inputs.read(declaration["proof"])
        self.tree = declaration["tree_sha256"]
        if (self.proof["checkpoint_path"] != str(self.path)
                or self.proof["control_tree_sha256"] != self.tree
                or not self.proof["status"].startswith("verified")):
            raise ValueError("source proof identity mismatch")
        self.checkpoint = reader.Checkpoint(self.path)
        self.files = self.proof["all_checkpoint_files"]
        if self.files != sorted(self.files, key=lambda row: row["path"]):
            raise ValueError("source file inventory must be sorted")
        if any(set(row) != {"path", "size", "sha256"}
               or not re.fullmatch(r"[0-9a-f]{64}", row["sha256"]) for row in self.files):
            raise ValueError("invalid prior file SHA inventory")
        if digest(canonical(self.files)) != self.tree:
            raise ValueError("source tree does not match prior file hashes")
        self.check()

    def check(self) -> None:
        identity = self.checkpoint.identity(hash_files=False)
        actual = identity["files"]
        if [(row["path"], row["size"]) for row in actual] != [
                (row["path"], row["size"]) for row in self.files]:
            raise ValueError("source file inventory changed")
        expected = self.proof["candidate_inventory_stat"]
        if set(expected) != {row["path"] for row in actual}:
            raise ValueError("source stat inventory differs")
        for row in actual:
            path = self.path / row["path"]
            if path.resolve() != path or path.is_symlink():
                raise ValueError("symlinked source file")
            previous = expected[row["path"]]
            if (row["size"] != previous["size"]
                    or any(row["stat"][key] != previous[key] for key in row["stat"])):
                raise ValueError("source stat continuity failed")


class Overlay:
    def __init__(self, plan_record: dict[str, str]):
        self.inputs = Inputs()
        self.plan_record = dict(plan_record)
        self.plan = self.inputs.read(plan_record)
        p = self.plan
        if p["schema_version"] != 1 or p["kind"] != FORMAT:
            raise ValueError("unsupported overlay plan")
        self.code = implementation_identity()
        if p["implementation_sha256"] != self.code:
            raise ValueError("overlay implementation differs")
        self.target = Source(p["physical_source"], self.inputs)
        self.draft = Source(p["draft_bypass"], self.inputs)
        if self.target.path == self.draft.path:
            raise ValueError("target and pinned draft must differ")
        self.parent = p["logical_parent"]
        # This version starts from a saved incumbent, never an unmaterialized overlay chain.
        if self.parent != {"path": str(self.target.path), "tree_sha256": self.target.tree}:
            raise ValueError("logical parent must match the saved physical source")
        self.spec = self.inputs.read(p["spec"])
        if (self.spec.get("source_checkpoint") != self.parent["path"]
                or self.spec.get("base_tree_sha256") != self.parent["tree_sha256"]):
            raise ValueError("spec is not bound to this logical parent")
        if self.spec.get("schema_version", 1) != 1 or self.spec.get("kind") is not None:
            raise ValueError("only sparse schema 1 is supported")
        edits = self.spec["edits"]
        if not isinstance(edits, list):
            raise ValueError("edits must be a list")
        for edit in edits:
            if (not PATTERN.fullmatch(edit["tensor"])
                    or set(edit) != {"tensor", "coordinates", "set_values"}):
                raise ValueError("only explicit O/MLP matrix set_values are supported")
        patches = editor.plan_edits(self.target.path, self.spec) if edits else []
        self.words: dict[str, dict[int, tuple[int, int]]] = {}
        for patch in patches:
            if patch["dtype"] != "BF16" or patch["size"] != 2:
                raise ValueError("only BF16 matrix words are supported")
            name = patch["tensor"]
            index = patch["selected_indices"][0]
            before = int.from_bytes(bytes.fromhex(patch["before_hex"]), "little")
            after = int.from_bytes(bytes.fromhex(patch["after_hex"]), "little")
            self.words.setdefault(name, {})[index] = (before, after)
        self.tp_size = p["tensor_parallel_size"]
        if type(self.tp_size) is not int or self.tp_size not in (1, 2):
            raise ValueError("only reviewed TP1/TP2 are supported")
        self.receipt_dir = exact_path(p["receipt_directory"])
        if any(self.receipt_dir.is_relative_to(s.path) for s in (self.target, self.draft)):
            raise ValueError("receipts cannot modify a source checkpoint")
        if not self.receipt_dir.is_dir():
            raise ValueError("owner must create a fresh receipt directory")
        self.logical_id = digest(canonical({
            "kind": FORMAT, "parent": self.parent, "spec_sha256": p["spec"]["sha256"],
            "implementation_sha256": self.code,
        }))
        self.seen: set[str] = set()
        self.finished = False
        self.check()

    def check(self) -> None:
        self.inputs.check()
        self.target.check()
        self.draft.check()
        if implementation_identity() != self.code:
            raise ValueError("overlay implementation changed")

    def role(self, path: str) -> str:
        resolved = exact_path(path)
        if resolved == self.target.path:
            return "target"
        if resolved == self.draft.path:
            return "draft_bypass"
        raise ValueError("unpinned checkpoint invocation")

    def iterator(self, weights: Iterable[tuple[str, Any]], path: str) -> Iterator[tuple[str, Any]]:
        import torch
        role = self.role(path)
        if role == "target" and (self.seen or self.finished):
            raise ValueError("one target iteration per loader required")
        self.check()
        for name, tensor in weights:
            if role == "target" and name in self.words:
                if name in self.seen:
                    raise ValueError("duplicate edited tensor")
                info = self.target.checkpoint.tensors[name]
                if (tensor.device.type != "cpu" or tensor.dtype != torch.bfloat16
                        or tuple(tensor.shape) != info.shape or not tensor.is_contiguous()):
                    raise ValueError("expected complete contiguous CPU BF16 tensor before TP")
                indices = sorted(self.words[name])
                index = torch.tensor(indices, dtype=torch.long)
                original = tensor.view(torch.int16).reshape(-1)[index].tolist()
                if [word & 65535 for word in original] != [
                        self.words[name][i][0] for i in indices]:
                    raise ValueError("iterator source words differ")
                # Never mutate mmap-backed source storage. No full-tensor FP32 conversion.
                tensor = tensor.clone()
                values = [self.words[name][i][1] for i in indices]
                signed = [word if word < 32768 else word - 65536 for word in values]
                tensor.view(torch.int16).reshape(-1)[index] = torch.tensor(signed, dtype=torch.int16)
                self.seen.add(name)
            yield name, tensor
        if role == "target":
            if self.seen != set(self.words):
                raise ValueError("edited tensors missing from loader iteration")
            self.finished = True
        self.check()

    def placement(self, name: str, index: int) -> tuple[int, str, tuple[int, int], tuple[int, int]]:
        """Reviewed Gemma4 MM dense loader mapping; owner rank, parameter, index, shape."""
        info = self.target.checkpoint.tensors[name]
        rows, cols = info.shape
        row, col = divmod(index, cols)
        mapped = name.replace("model.language_model.", "language_model.model.", 1)
        if ".gate_proj." in name or ".up_proj." in name:
            if rows % self.tp_size:
                raise ValueError("nondivisible merged-column TP shape")
            width = rows // self.tp_size
            rank = row // width
            local_row = row % width + (width if ".up_proj." in name else 0)
            mapped = mapped.replace(".gate_proj.", ".gate_up_proj.").replace(".up_proj.", ".gate_up_proj.")
            return rank, mapped, (local_row, col), (2 * width, cols)
        if cols % self.tp_size:
            raise ValueError("nondivisible row-parallel TP shape")
        width = cols // self.tp_size
        return col // width, mapped, (row, col % width), (rows, width)

    def rank_proof(self, model: Any, rank: int) -> dict[str, Any]:
        import torch
        if not self.finished or not 0 <= rank < self.tp_size:
            raise ValueError("post-TP proof requires completed target iteration and valid rank")
        params = dict(model.named_parameters())
        records = []
        for name, words in sorted(self.words.items()):
            selected = [(index, self.placement(name, index)) for index in sorted(words)]
            selected = [(index, place) for index, place in selected if place[0] == rank]
            if not selected:
                continue
            parameter, shape = selected[0][1][1], selected[0][1][3]
            tensor = params.get(parameter)
            if tensor is None or tensor.dtype != torch.bfloat16 or tuple(tensor.shape) != shape:
                raise ValueError("loaded parameter missing or incompatible with TP mapping")
            # One selected-word transfer per tensor, not one GPU synchronization per word.
            rows = torch.tensor([place[2][0] for _, place in selected], device=tensor.device)
            cols = torch.tensor([place[2][1] for _, place in selected], device=tensor.device)
            values = tensor.detach().view(torch.int16)[rows, cols].cpu().tolist()
            raw = struct.pack("<" + "H" * len(values), *(value & 65535 for value in values))
            records.append({"tensor": name, "indices": [index for index, _ in selected],
                            "actual_words_hex": raw.hex()})
        self.check()
        return {"rank": rank, "tensor_parallel_size": self.tp_size,
                "logical_overlay_sha256": self.logical_id, "physical_tree_sha256": self.target.tree,
                "plan_sha256": self.plan_record["sha256"],
                "method": "post_load_post_tp_selected_bf16_readback", "records": records}

    def aggregate(self, proofs: list[dict[str, Any]]) -> dict[str, Any]:
        """All ranks must agree; each selected global coordinate has exactly one owner."""
        if len(proofs) != self.tp_size or {p.get("rank") for p in proofs} != set(range(self.tp_size)):
            raise ValueError("missing or duplicate TP rank proof")
        covered: set[tuple[str, int]] = set()
        for proof in proofs:
            if (proof.get("error") or proof.get("logical_overlay_sha256") != self.logical_id
                    or proof.get("physical_tree_sha256") != self.target.tree
                    or proof.get("plan_sha256") != self.plan_record["sha256"]
                    or proof.get("tensor_parallel_size") != self.tp_size
                    or proof.get("method") != "post_load_post_tp_selected_bf16_readback"):
                raise ValueError("rank proof identity/error mismatch")
            for record in proof["records"]:
                name = record["tensor"]
                indices = record["indices"]
                raw = bytes.fromhex(record["actual_words_hex"])
                if len(raw) != 2 * len(indices):
                    raise ValueError("rank proof word length mismatch")
                for index, (actual,) in zip(indices, struct.iter_unpack("<H", raw), strict=True):
                    key = name, index
                    if (name not in self.words or index not in self.words[name] or key in covered
                            or self.placement(name, index)[0] != proof["rank"]
                            or self.words[name][index][1] != actual):
                        raise ValueError("post-TP selected word mismatch/ownership error")
                    covered.add(key)
        expected = {(name, index) for name, words in self.words.items() for index in words}
        if covered != expected:
            raise ValueError("incomplete all-rank selected-word coverage")
        self.check()
        return {"schema_version": 1, "status": "verified_selected_words_all_tp_ranks",
                "logical_overlay_sha256": self.logical_id, "logical_parent": self.parent,
                "physical_checkpoint_tree_sha256": self.target.tree,
                "physical_checkpoint_is_overlay_candidate": False,
                "source_proof": "prior_full_file_sha256_plus_exact_current_stat",
                "fresh_full_checkpoint_hash": False, "empty_overlay": not self.words,
                "selected_coordinates": len(expected), "selected_tensors": len(self.words),
                "storage_bytes_exact": True, "scope": "selected coordinates only",
                "rank_proofs": sorted(proofs, key=lambda p: p["rank"]),
                "plan": self.plan_record, "draft_tree_sha256": self.draft.tree,
                "implementation_sha256": self.code}
