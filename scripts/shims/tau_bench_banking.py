from __future__ import annotations

import os
from typing import Any


OLD_EMBEDDING_MODEL = "text-embedding-3-large"


def configure_embedding_model(
    model_id: str,
    *,
    retrieval_module: Any | None = None,
    embeddings_cache_module: Any | None = None,
) -> None:
    if not model_id:
        raise RuntimeError("KAIRYU_EMBEDDING_MODEL must not be empty")

    if retrieval_module is None:
        from tau2.domains.banking_knowledge import retrieval as retrieval_module
    if embeddings_cache_module is None:
        from tau2.knowledge import embeddings_cache as embeddings_cache_module

    try:
        dense = retrieval_module.RETRIEVAL_VARIANTS["alltools"].kb_search_dense
    except (AttributeError, KeyError) as error:
        raise RuntimeError("pinned tau2 alltools dense retrieval layout changed") from error
    if dense is None or dense.embedder_type != "openai":
        raise RuntimeError("pinned tau2 alltools no longer uses OpenAI dense retrieval")
    if dense.embedder_model != OLD_EMBEDDING_MODEL:
        raise RuntimeError(
            "pinned tau2 alltools dense retrieval model no longer matches the shim"
        )
    dense.embedder_model = model_id

    original_selector = (
        embeddings_cache_module.get_unique_embedder_configs_for_retrieval_configs
    )

    def select_embedder_configs(
        retrieval_config_names: list[str],
        retrieval_config_kwargs: dict[str, Any] | None = None,
    ) -> list[tuple[str, dict[str, Any]]]:
        configs = original_selector(
            retrieval_config_names,
            retrieval_config_kwargs,
        )
        if not any(
            name in {"alltools", "AllTools"} for name in retrieval_config_names
        ):
            return configs

        rewritten: list[tuple[str, dict[str, Any]]] = []
        replaced = False
        for embedder_type, params in configs:
            if (
                embedder_type == "openai"
                and params.get("model") == OLD_EMBEDDING_MODEL
            ):
                if not replaced:
                    rewritten.append(("openai", {"model": model_id}))
                    replaced = True
                continue
            rewritten.append((embedder_type, params))
        if not replaced:
            raise RuntimeError("pinned tau2 alltools cache warmup layout changed")
        return rewritten

    embeddings_cache_module.get_unique_embedder_configs_for_retrieval_configs = (
        select_embedder_configs
    )


def main() -> int:
    model_id = os.environ.get("KAIRYU_EMBEDDING_MODEL", "")
    configure_embedding_model(model_id)

    from tau2.cli import main as tau2_main

    result = tau2_main()
    return 0 if result is None else int(result)


if __name__ == "__main__":
    raise SystemExit(main())
