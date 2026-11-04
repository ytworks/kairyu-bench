"""Math: GSM8K-Platinum, MATH-500, AIME 2026, HMMT Feb 2026 and MGSM.

Prompts: GSM8K-Platinum's own (reasoning-model variant), lighteval's MATH-500,
MathArena's AIME/HMMT instructions, simple-evals' MGSM. Expressions are graded
with Math-Verify; GSM8K-Platinum and MGSM keep their official number parsers.
"""

from __future__ import annotations

import multiprocessing
import re
from concurrent.futures import ProcessPoolExecutor
from typing import Any

from kairyu_bench.wsp.catalog import SOURCE_BY_NAME
from kairyu_bench.wsp.manifest import Item
from kairyu_bench.wsp.sources import Adapter, Loader, Outcome, Row, Scored, register, run_chat, score_each


def _chat(row: Row) -> list[dict[str, str]]:
    return [{"role": "user", "content": row["prompt"]}]


# MadryLab/platinum-benchmarks 8fd2f82e63c49ea1cca4266f4dded82b7ddbcb55
# src/run_gsm8k_platinum.py: reasoning models get the prompt without
# "Think step-by-step. "; src/utils.py parse_fn (math strategy).
PLATINUM_TEMPLATE = """Solve the following math word problem.

{question}

Think step-by-step. Then, provide the final answer as a single integer in the format "Answer: XXX" with no extra formatting."""


def _platinum_parse(output: str) -> str:
    tex_pattern = r"\\boxed\{(\\text\{)?([^\\{}]+)\}"
    if "answer:" in output.lower().replace("*", ""):
        answer_section = output.lower().split("answer: ")[-1]
        if re.search(tex_pattern, answer_section):
            output = "Answer: " + re.search(tex_pattern, answer_section).group(2)  # type: ignore[union-attr]
    elif re.search(tex_pattern, output):
        output = "Answer: " + re.search(tex_pattern, output).group(2)  # type: ignore[union-attr]
    else:
        output = "Answer: " + output.strip("\n").split("\n")[-1].lower()
    found = re.search(
        r"-?[0-9.]*[0-9]",
        output.replace("*", "").replace("#", "").lower().split("answer: ")[-1].replace(",", ""),
    )
    if found is None:
        raise ValueError("no number")
    return re.sub(r"\.0+$", "", found.group())


def _platinum_score(row: Row, sample: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    try:
        prediction = _platinum_parse(sample["content"])
        return float(float(row["target"]) == float(prediction)), {"extracted": prediction}
    except (ValueError, AttributeError):
        return 0.0, {"extracted": None}


def gsm8k_group(question: str) -> str:
    """GSM8K problem identity across GSM8K-Platinum and MGSM's English text."""

    import hashlib

    normalized = re.sub(r"\s+", " ", question.strip().lower())
    return "gsm8k:" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def _platinum(loader: Loader) -> dict[str, Row]:
    spec = SOURCE_BY_NAME["gsm8k-platinum"]
    rows = loader.dataset(spec.dataset or "", config="main", split="test", revision=spec.revision)
    return {
        f"gsm8k-platinum:{index:04d}": {
            "prompt": PLATINUM_TEMPLATE.format(question=row["question"]).replace("Think step-by-step. ", ""),
            "target": row["answer"].split("\n#### ")[-1].replace(",", ""),
            "status": row["cleaning_status"],
            "group": gsm8k_group(row["question"]),
        }
        for index, row in enumerate(rows)
    }


register(
    "gsm8k-platinum",
    Adapter(
        load=_platinum,
        stratum=lambda row: str(row["status"]),
        group=lambda row: str(row["group"]),
        execute=lambda context, item, row: run_chat(context, item, _chat(row)),
        score=score_each(_platinum_score),
    ),
)


# huggingface/lighteval b97a6533bec22c4184b88526366fdf5d73d502c9 tasks/math_500.py.
MATH_500_TEMPLATE = (
    'Solve the following problem. The final line of your response MUST be of the following format:\n'
    '"ANSWER: $ANSWER" (without quotes) where $ANSWER is the final answer. Think step by step before answering.\n\n'
    "{prompt}"
)
# eth-sri/matharena cec19dfe83e2881a8bfcf640f16d0d29718b033a
# configs/competitions/{aime/aime_2026,hmmt/hmmt_feb_2026}.yaml: instruction + "\n\n{problem}".
AIME_INSTRUCTION = "Put your final answer within \\boxed{}.\nThe answer is an integer between 0 and 999 inclusive."
HMMT_INSTRUCTION = "Put your final answer within \\boxed{}."


def _math_500(loader: Loader) -> dict[str, Row]:
    spec = SOURCE_BY_NAME["math-500"]
    rows = loader.dataset(spec.dataset or "", split="test", revision=spec.revision)
    return {
        f"math-500:{row['unique_id']}": {
            "prompt": MATH_500_TEMPLATE.format(prompt=row["problem"]),
            "answer": row["answer"],
            "stratum": f"level {row['level']}",
        }
        for row in rows
    }


def _matharena(source: str, instruction: str):  # type: ignore[no-untyped-def]
    spec = SOURCE_BY_NAME[source]

    def load(loader: Loader) -> dict[str, Row]:
        rows = loader.dataset(spec.dataset or "", split="train", revision=spec.revision)
        return {
            f"{source}:{row['problem_idx']}": {
                "prompt": instruction + "\n\n" + row["problem"],
                "answer": str(row["answer"]),
                "stratum": "all",
            }
            for row in rows
        }

    return load


def _verify_one(gold: str, response: str) -> tuple[bool, str | None]:
    """Math-Verify in a worker process (its timeouts need the main thread)."""

    from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify

    gold_parsed = parse("$" + gold + "$", [LatexExtractionConfig()])
    predicted = parse(response, [LatexExtractionConfig(boxed_match_priority=0), ExprExtractionConfig()])
    if not gold_parsed or not predicted:
        return False, None
    return bool(verify(gold_parsed, predicted)), str(predicted[-1])


def _score_verified(entries: list[tuple[Item, Row, Outcome]], loader: Loader) -> list[Scored]:
    jobs: list[tuple[int, int, str, str]] = []
    for position, (_, row, outcome) in enumerate(entries):
        for sample_index, sample in enumerate(outcome["samples"]):
            if sample.get("finish_reason") != "length":
                jobs.append((position, sample_index, row["answer"], sample.get("content") or ""))
    verdicts: dict[tuple[int, int], tuple[bool, str | None]] = {}
    if jobs:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=8, mp_context=context) as pool:
            results = pool.map(_verify_one, [job[2] for job in jobs], [job[3] for job in jobs], chunksize=8)
            for job, verdict in zip(jobs, results, strict=True):
                verdicts[(job[0], job[1])] = verdict
    scored: list[Scored] = []
    for position, (_, _, outcome) in enumerate(entries):
        values: list[float] = []
        details: list[dict[str, Any]] = []
        for sample_index, sample in enumerate(outcome["samples"]):
            found = verdicts.get((position, sample_index))
            if found is None:
                values.append(0.0)
                details.append({"truncated": sample.get("finish_reason") == "length"})
            else:
                values.append(float(found[0]))
                details.append({"extracted": found[1]})
        scored.append(Scored(sum(values) / len(values), tuple(values), {"samples": details}))
    return scored


for _source, _load in (
    ("math-500", _math_500),
    ("aime-2026", _matharena("aime-2026", AIME_INSTRUCTION)),
    ("hmmt-feb-2026", _matharena("hmmt-feb-2026", HMMT_INSTRUCTION)),
):
    register(
        _source,
        Adapter(
            load=_load,
            stratum=lambda row: str(row["stratum"]),
            execute=lambda context, item, row: run_chat(context, item, _chat(row)),
            score=_score_verified,
        ),
    )


# openai/simple-evals 652c89d0ca9df547706735883097e9537d40dc47 mgsm_eval.py
# (verbatim, including upstream's missing "こ" in the Japanese instruction).
MGSM_INSTRUCTIONS = {
    "en": """Solve this math problem. Give the reasoning steps before giving the final answer on the last line by itself in the format of "Answer:". Do not add anything other than the integer answer after "Answer:".

{input}""",
    "bn": """এই গণিতের সমস্যাটি সমাধান করুন। চূড়ান্ত উত্তর দেওয়ার আগে যুক্তিসম্পন্ন পদক্ষেপ প্রদান করুন। চূড়ান্ত উত্তরটি একক সংখ্যা হিসাবে "উত্তর:" এর পরে শেষ লাইনে দিন। "উত্তর:" এর পরে অন্য কিছু যুক্ত করবেন না।.

{input}""",
    "de": """Löse dieses Mathematikproblem. Gib die Schritte zur Begründung an, bevor du die endgültige Antwort in der letzten Zeile alleine im Format "Antwort:" gibst. Füge nichts anderes als die ganzzahlige Antwort nach "Antwort:" hinzu.

{input}""",
    "es": """Resuelve este problema matemático. Proporciona los pasos de razonamiento antes de dar la respuesta final en la última línea por sí misma en el formato de "Respuesta:". No añadas nada más que la respuesta entera después de "Respuesta:".

{input}""",
    "fr": """Résolvez ce problème de mathématiques. Donnez les étapes de raisonnement avant de fournir la réponse finale sur la dernière ligne elle-même dans le format de "Réponse:". N'ajoutez rien d'autre que la réponse entière après "Réponse:".

{input}""",
    "ja": """の数学の問題を解いてください。最終的な答えを出す前に、解答の推論過程を記述してください。そして最後の行には "答え:" の形式で答えを記述し、その後には整数の答え以外何も追加しないでください。

{input}""",
    "ru": """Решите эту математическую задачу. Объясните шаги рассуждения перед тем, как дать окончательный ответ в последней строке сам по себе в формате "Ответ:". Не добавляйте ничего, кроме целочисленного ответа после "Ответ:".

{input}""",
    "sw": """Suluhisha tatizo hili la hesabu. Toa hatua za mantiki kabla ya kutoa jibu la mwisho kwenye mstari wa mwisho peke yake katika muundo wa "Jibu:". Usiongeze chochote kingine isipokuwa jibu la integer baada ya "Jibu:".

{input}""",
    "te": """ఈ గణిత సమస్యను పరిష్కరించండి. చివరి సమాధానాన్ని ఇవ్వదానికి ముందు తర్కాత్మక అదుగులను ఇవ్వండి. చివరి పంక్తిలో మాత్రమే 'సమాధానం:' అనే ఆకారంలో చివరి సమాధానాద్ని ఇవ్వండి సమాధానం: తర్వాత పూర్ణాంక సమాధానానికి తప్పించి ఎదేనా చేర్చవద్దు.

{input}""",
    "th": """แก้ปัญหาคณิตศาสตร์นี้ ให้ให้ขั้นตอนการใช้เหตุผลก่อนที่จะให้คำตอบสุดท้ายในบรรทัดสุดท้ายโดยอยู่ในรูปแบบ "คำตอบ:" ไม่ควรเพิ่มอะไรนอกจากคำตอบที่เป็นจำนวนเต็มหลังจาก "คำตอบ:"

{input}""",
    "zh": """解决这个数学问题。在最后一行给出答案前，请提供推理步骤。最后一行应该以 "答案: " 的形式独立给出答案。在 "答案：" 后不要添加除整数答案之外的任何内容。

{input}""",
}
MGSM_ANSWER_PREFIX = {
    "en": "Answer",
    "bn": "উত্তর",
    "de": "Antwort",
    "es": "Respuesta",
    "fr": "Réponse",
    "ja": "答え",
    "ru": "Ответ",
    "sw": "Jibu",
    "te": "సమాధానం",
    "th": "คำตอบ",
    "zh": "答案",
}


def _mgsm_parse(answer: str, answer_prefix: str) -> str:
    if answer_prefix not in answer:
        return ""
    answer_text = answer.split(answer_prefix)[-1].strip()
    numbers = re.findall(r"\d+\.?\d*", answer_text.replace(",", ""))
    return numbers[-1].rstrip(".") if numbers else ""


def _mgsm_score(target: str, prediction: str) -> bool:
    if "." in prediction:
        prediction = prediction.rstrip("0").rstrip(".")
    return target.replace(",", "") == prediction.replace(",", "")


def _mgsm(loader: Loader) -> dict[str, Row]:
    spec = SOURCE_BY_NAME["mgsm"]
    loaded: dict[str, Row] = {}
    english = loader.dataset(spec.dataset or "", config="en", split="test", revision=spec.revision)
    groups = [gsm8k_group(row["question"]) for row in english]
    for language in MGSM_INSTRUCTIONS:
        rows = loader.dataset(spec.dataset or "", config=language, split="test", revision=spec.revision)
        for index, row in enumerate(rows):
            loaded[f"mgsm:{language}:{index:03d}"] = {
                "prompt": MGSM_INSTRUCTIONS[language].format(input=row["question"].strip()),
                "target": str(row["answer_number"]),
                "language": language,
                "group": groups[index],
            }
    return loaded


def _mgsm_sample(row: Row, sample: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    prediction = _mgsm_parse(sample["content"], MGSM_ANSWER_PREFIX[row["language"]])
    return float(_mgsm_score(row["target"], prediction)), {"extracted": prediction}


register(
    "mgsm",
    Adapter(
        load=_mgsm,
        stratum=lambda row: str(row["language"]),
        group=lambda row: str(row["group"]),
        execute=lambda context, item, row: run_chat(context, item, _chat(row)),
        score=score_each(_mgsm_sample),
    ),
)
