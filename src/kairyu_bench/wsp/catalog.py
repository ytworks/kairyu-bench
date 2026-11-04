"""Capability areas and the public sources that measure them.

Every tier measures every area; tiers differ only in how many items each
source contributes and how long one item may run. Scoring never uses an LLM
judge: an edited model must not grade itself.
"""

from __future__ import annotations

from dataclasses import dataclass

TIERS = (10, 30, 60)
SPLITS = ("dev", "test")
# Request kinds: a chat completion, a prompt-logprob scan, an agent task.
KINDS = ("chat", "logprob", "agent")


@dataclass(frozen=True)
class Area:
    key: str
    label: str


@dataclass(frozen=True)
class Source:
    name: str
    area: str | None  # None: a diagnostic outside the composite score
    kind: str
    dataset: str | None
    revision: str | None
    license: str
    samples: int = 1
    first_tier: int = 10


AREAS = (
    Area("knowledge", "一般知識"),
    Area("expert", "専門知識・科学推論"),
    Area("facts", "事実の記憶"),
    Area("math", "数学"),
    Area("code", "コード生成"),
    Area("instructions", "指示追従"),
    Area("tools", "関数呼び出し"),
    Area("agents", "エージェント作業"),
    Area("long-context", "長文脈"),
    Area("reasoning", "論理推論"),
    Area("commonsense", "読解・常識"),
    Area("multilingual", "多言語・日本語"),
    Area("vision", "画像理解"),
    Area("safety", "安全性"),
)

SOURCES = (
    Source("mmlu-pro", "knowledge", "chat", "TIGER-Lab/MMLU-Pro",
           "b189ec765aa7ed75c8acfea42df31fdae71f97be", "MIT"),
    Source("gpqa-diamond", "expert", "chat", "Idavidrein/gpqa",
           "633f5ee89ab8ad4522a9f850766b73f62147ffdd", "CC-BY-4.0 (gated)"),
    Source("triviaqa", "facts", "chat", "mandarjoshi/trivia_qa",
           "0f7faf33a3908546c6fd5b73a660e0f8ff173c2f", "unspecified on the dataset card"),
    Source("gsm8k-platinum", "math", "chat", "madrylab/gsm8k-platinum",
           "e762492455a1cf7967de89f05b6bef72fc713b66", "MIT"),
    Source("math-500", "math", "chat", "HuggingFaceH4/MATH-500",
           "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be", "unspecified on the dataset card"),
    Source("aime-2026", "math", "chat", "MathArena/aime_2026",
           "d61f486805306e6e6b5913504c07763a99e1b112", "CC-BY-NC-SA-4.0", samples=4),
    Source("hmmt-feb-2026", "math", "chat", "MathArena/hmmt_feb_2026",
           "589a53fe152febc0f4f99f3e561007bf8da8e664", "CC-BY-NC-SA-4.0", samples=4),
    Source("humaneval-plus", "code", "chat", "evalplus/humanevalplus",
           "d32357cf319e50e9c8d8dab5ea876c72b0fd321b", "Apache-2.0"),
    Source("mbpp-plus", "code", "chat", "evalplus/mbppplus",
           "b2d74c91837c3f2a20c1299ae98133cbe7cfa077", "Apache-2.0"),
    Source("livecodebench", "code", "chat", "livecodebench/code_generation_lite",
           "0fe84c3912ea0c4d4a78037083943e8f0c4dd505", "CC (dataset card)"),
    Source("scicode", "code", "chat", "SciCode1/SciCode",
           "4510f6a6aa27c43fad7b43da2c59602a86e88480", "Apache-2.0", first_tier=30),
    Source("ifeval", "instructions", "chat", "google/IFEval",
           "966cd89545d6b6acfd7638bc708b98261ca58e84", "Apache-2.0"),
    Source("ifbench", "instructions", "chat", "allenai/IFBench_test",
           "ed7c12452594b7028b15807b86d3938150b663ae", "ODC-BY-1.0"),
    Source("bfcl", "tools", "chat", "gorilla-llm/Berkeley-Function-Calling-Leaderboard",
           "61fc0608cfd831fcfbbaa676ebdfef0ed963eeda", "Apache-2.0"),
    Source("terminal-bench", "agents", "agent", "terminal-bench/terminal-bench-2-1",
           "2.1", "per task (repository)"),
    Source("swe-bench-pro", "agents", "agent", "ScaleAI/SWE-bench_Pro",
           "7ab5114912baf22bb098818e604c02fe7ad2c11f", "per repository", first_tier=30),
    Source("ruler", "long-context", "chat", None, None, "Apache-2.0 (generator)"),
    Source("mrcr-v2", "long-context", "chat", "openai/mrcr",
           "f4c69fae7cf81f7ca26b9fee34b392a50f6b8a1d", "MIT"),
    Source("longbench-v2", "long-context", "chat", "THUDM/LongBench-v2",
           "2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9", "Apache-2.0", first_tier=30),
    Source("bbeh", "reasoning", "chat", "BBEH/bbeh",
           "08e07a803851822c04399782ece3c4a07ce419f9", "Apache-2.0"),
    Source("musr", "reasoning", "chat", "TAUR-Lab/MuSR",
           "7c365b439a222150f317764d4f16ae6c96d7d94a", "CC-BY-4.0"),
    Source("hellaswag", "commonsense", "logprob", "Rowan/hellaswag",
           "218ec52e09a7e7462a5400043bb9a69a41d06b76", "unspecified on the dataset card"),
    Source("arc-challenge", "commonsense", "logprob", "allenai/ai2_arc",
           "210d026faf9955653af8916fad021475a3f00453", "CC-BY-SA-4.0"),
    Source("winogrande", "commonsense", "logprob", "allenai/winogrande",
           "01e74176c63542e6b0bcb004dcdea22d94fb67b5", "unspecified on the dataset card"),
    Source("mgsm", "multilingual", "chat", "juletxara/mgsm",
           "b2f13d426afe3be8d69a7e739b36724db8b66bbc", "CC-BY-SA-4.0"),
    Source("mmlu-prox-ja", "multilingual", "chat", "li-lab/MMLU-ProX",
           "8e6106a6c6ce1c5027e66cc338143cf997b2aa09", "MIT"),
    Source("jamc-qa", "multilingual", "chat", "sbintuitions/JamC-QA",
           "ec2f8a34b1cfb9dcd8c50583eb688ef2ee3fb671", "CC-BY-SA-4.0"),
    Source("mmmu-pro", "vision", "chat", "MMMU/MMMU_Pro",
           "0d7426df4ccb3d8704a992fe6850abee7c262127", "Apache-2.0"),
    Source("xstest", "safety", "chat", None, None, "CC-BY-4.0"),
    # Next-token distribution probes against the base model (diagnostics).
    Source("probe-wikitext", None, "logprob", "Salesforce/wikitext",
           "b08601e04326c79dfdd32d625aee71d232d685c3", "CC-BY-SA-3.0"),
    Source("probe-wikipedia-ja", None, "logprob", "wikimedia/wikipedia",
           "b04c8d1ceb2f5cd4588862100d08de323dccfbaa", "CC-BY-SA-3.0"),
    Source("probe-code", None, "logprob", "evalplus/mbppplus",
           "b2d74c91837c3f2a20c1299ae98133cbe7cfa077", "Apache-2.0"),
)

AREA_KEYS = tuple(area.key for area in AREAS)
SOURCE_BY_NAME = {source.name: source for source in SOURCES}

assert len(SOURCE_BY_NAME) == len(SOURCES), "duplicate source name"
assert all(source.kind in KINDS for source in SOURCES)
assert all(source.first_tier in TIERS for source in SOURCES)
assert all(source.area is None or source.area in AREA_KEYS for source in SOURCES)
# Every area is measured in every tier.
assert all(
    any(source.area == area and source.first_tier == TIERS[0] for source in SOURCES)
    for area in AREA_KEYS
)
