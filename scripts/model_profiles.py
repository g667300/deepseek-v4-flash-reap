#!/usr/bin/env python3
"""Per-model knowledge, kept out of the scoring and surgery machinery.

A profile answers the handful of questions the pipeline actually has about a
checkpoint: what a tensor name means, which quantization scheme its weights
use, which blocks may be pruned, and how the config records the expert count.
This repository ships one profile, for DeepSeek-V4; the split exists so a
second model (a different MoE family) is a new ``ModelProfile`` instance
rather than edits to either pass itself.

Terminology: a **block** is the tensor-name prefix that owns one router and one
set of experts -- ``layers.12`` or ``mtp.0`` for DeepSeek-V4. Transformer
layers and MTP blocks are both blocks, which is what lets the same code prune
or drop either.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ModelProfile:
    """Everything the REAP stages need to know about one checkpoint family."""

    name: str
    quant_scheme: str
    """Key understood by :func:`quant.dequantize_linear` (``nvfp4`` / ``deepseek``)."""

    expert_re: re.Pattern
    """Matches an expert tensor. Groups: (block, expert index, remainder)."""

    expert_fmt: str
    """Rebuilds an expert tensor name from ``block``/``expert``/``rest``."""

    router_re: re.Pattern
    """Matches a router tensor whose dim-0 is indexed by expert. Group 1: block."""

    block_re: re.Pattern
    """Matches any per-block tensor. Group 1: block prefix (no trailing dot)."""

    layer_fmt: str
    """Rebuilds the block prefix of transformer layer ``L``."""

    layer_re: re.Pattern
    """Matches a *transformer layer* block prefix. Group 1: layer index."""

    config_expert_keys: tuple[str, ...] = ("n_routed_experts", "num_experts")
    index_name: str = "model.safetensors.index.json"

    expert_id_valued_re: re.Pattern | None = None
    """Tensors holding expert *ids as values* (DeepSeek's ``tid2eid``).

    Slicing rows would silently corrupt these, so blocks that own one are
    refused for pruning rather than handled.
    """

    mtp_re: re.Pattern | None = None
    """Matches MTP block prefixes outright (DeepSeek-V4's ``mtp.N``)."""

    hash_layer_count_keys: tuple[str, ...] = ()
    """Config keys giving a count of leading layers that route by hash."""

    aux_dirs_to_copy: tuple[str, ...] = field(default=())
    """Sub-directories copied verbatim into the output checkpoint."""

    chat_encoder: tuple[str, str] | None = None
    """``(path relative to the checkpoint, function name)`` for models that ship
    their own prompt encoder instead of an HF ``chat_template``.

    DeepSeek-V4 is one: ``tokenizer_config.json`` has no template at all, and
    the real format (BOS, ``<|User|>`` / ``<|Assistant|>``, ``<think>`` blocks)
    lives in ``encoding/encoding_dsv4.py``. Calibration text has to go through
    it or the calibration distribution will not match how the model is served.
    """

    # -- derived helpers ---------------------------------------------------

    def chat_renderer(self, source_dir: Path):
        """Return ``f(messages, **kwargs) -> str``, or None to use the template."""
        if self.chat_encoder is None:
            return None
        import importlib.util

        rel, func = self.chat_encoder
        path = Path(source_dir) / rel
        if not path.exists():
            raise FileNotFoundError(
                f"{self.name} renders prompts with {rel}, which is missing from {source_dir}"
            )
        spec = importlib.util.spec_from_file_location(f"_chat_encoder_{self.name}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return getattr(module, func)

    def block_of_layer(self, layer: int) -> str:
        return self.layer_fmt.format(layer=layer)

    def layer_of_block(self, block: str) -> int | None:
        m = self.layer_re.fullmatch(block)
        return int(m.group(1)) if m else None

    def expert_name(self, block: str, expert: int, rest: str) -> str:
        return self.expert_fmt.format(block=block, expert=expert, rest=rest)

    def blocks_in(self, names) -> set[str]:
        out = set()
        for name in names:
            m = self.block_re.match(name)
            if m:
                out.add(m.group(1))
        return out

    def mtp_blocks(self, all_blocks: set[str], pruned: set[str], weight_map) -> list[str]:
        """Blocks that hold an MTP head rather than a served transformer layer.

        DeepSeek-V4 marks these by name (``mtp.N``); this profile relies on
        ``mtp_re`` being set rather than a name-independent fallback.
        """
        if self.mtp_re is not None:
            return sorted(b for b in all_blocks if self.mtp_re.fullmatch(b))
        routed = {
            m.group(1) for name in weight_map if (m := self.router_re.match(name))
        }
        return sorted(routed - pruned)

    def hash_layers(self, config: dict) -> list[int]:
        """Leading layers whose expert choice is a frozen token-id lookup."""
        for key in self.hash_layer_count_keys:
            if key in config:
                return list(range(int(config[key])))
        types = config.get("mlp_layer_types")
        if types:
            return [i for i, t in enumerate(types) if t == "hash_moe"]
        return []

    def unprunable_blocks(self, config: dict) -> list[str]:
        return [self.block_of_layer(L) for L in self.hash_layers(config)]

    def patch_config(self, cfg: dict, n_new: int, dropped: set[str]) -> None:
        """Update the expert count and anything invalidated by dropped blocks."""
        for key in self.config_expert_keys:
            if key in cfg:
                cfg[key] = n_new
        if dropped:
            if "num_nextn_predict_layers" in cfg:
                cfg["num_nextn_predict_layers"] = 0
            qc = cfg.get("quantization_config", {})
            if isinstance(qc, dict) and "ignore" in qc:
                qc["ignore"] = _drop_ignore_patterns(qc["ignore"], dropped)


def _drop_ignore_patterns(patterns: list[str], dropped: set[str]) -> list[str]:
    """Remove modelopt ignore globs that only ever matched a dropped block."""
    prefixes = tuple(f"{b}." for b in dropped) + tuple(f"{b}*" for b in dropped)
    return [p for p in patterns if not p.startswith(prefixes)]


# --------------------------------------------------------------------------
# profiles
# --------------------------------------------------------------------------

DEEPSEEK_V4 = ModelProfile(
    name="deepseek_v4",
    quant_scheme="deepseek",
    expert_re=re.compile(r"^((?:layers|mtp)\.\d+)\.ffn\.experts\.(\d+)\.(.+)$"),
    expert_fmt="{block}.ffn.experts.{expert}.{rest}",
    router_re=re.compile(r"^((?:layers|mtp)\.\d+)\.ffn\.gate\.(?:weight|bias)$"),
    block_re=re.compile(r"^((?:layers|mtp)\.\d+)\."),
    layer_fmt="layers.{layer}",
    layer_re=re.compile(r"^layers\.(\d+)$"),
    expert_id_valued_re=re.compile(r"\.ffn\.gate\.tid2eid$"),
    mtp_re=re.compile(r"^mtp\.\d+$"),
    hash_layer_count_keys=("num_hash_layers",),
    aux_dirs_to_copy=("inference", "encoding"),
    chat_encoder=("encoding/encoding_dsv4.py", "encode_messages"),
)

PROFILES = {p.name: p for p in (DEEPSEEK_V4,)}

_ARCH_TO_PROFILE = {
    "DeepseekV4ForCausalLM": DEEPSEEK_V4,
}


def detect(checkpoint: Path, config: dict | None = None) -> ModelProfile:
    """Pick a profile from a checkpoint's ``config.json``."""
    if config is None:
        import json

        config = json.loads((Path(checkpoint) / "config.json").read_text())
    for arch in config.get("architectures") or []:
        if arch in _ARCH_TO_PROFILE:
            return _ARCH_TO_PROFILE[arch]
    model_type = config.get("model_type")
    for profile in PROFILES.values():
        if profile.name == model_type:
            return profile
    raise ValueError(
        f"no REAP profile for architectures={config.get('architectures')} "
        f"model_type={model_type!r}; add one to scripts/model_profiles.py"
    )


def detect_optional(checkpoint: Path) -> ModelProfile | None:
    """Like :func:`detect`, but returns None instead of raising.

    For callers (calibration) that only need a profile if one happens to apply
    and are handed plain tokenizer directories the rest of the time.
    """
    import json

    path = Path(checkpoint) / "config.json"
    if not path.exists():
        return None
    try:
        return detect(checkpoint, json.loads(path.read_text()))
    except ValueError:
        return None


def get(name: str) -> ModelProfile:
    if name not in PROFILES:
        raise ValueError(f"unknown profile {name!r}; have {sorted(PROFILES)}")
    return PROFILES[name]
