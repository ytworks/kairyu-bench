"""Multiple-choice knowledge sources answered in chat, each with its official
zero-shot prompt and answer extraction."""

from __future__ import annotations

import re
import string

from kairyu_bench.direct import deterministic_gpqa_choices
from kairyu_bench.wsp.catalog import SOURCE_BY_NAME
from kairyu_bench.wsp.sources import Adapter, Loader, Row, register, run_chat, score_each

# openai/simple-evals 652c89d0ca9df547706735883097e9537d40dc47 common.py.
SIMPLE_EVALS_MULTICHOICE = """
Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

{Question}

A) {A}
B) {B}
C) {C}
D) {D}
""".strip()
SIMPLE_EVALS_ANSWER = r"(?i)Answer[ \t]*:[ \t]*\$?([A-D])\$?"


def _chat(row: Row) -> list[dict[str, str]]:
    return [{"role": "user", "content": row["prompt"]}]


def _register(name: str, load, stratum, extract, group=None) -> None:  # type: ignore[no-untyped-def]
    register(
        name,
        Adapter(
            load=load,
            stratum=stratum,
            group=group,
            execute=lambda context, item, row: run_chat(context, item, _chat(row)),
            score=score_each(
                lambda row, sample: (
                    float(extract(sample["content"]) == row["expected"]),
                    {"extracted": extract(sample["content"])},
                )
            ),
        ),
    )


def _gpqa(loader: Loader) -> dict[str, Row]:
    # simple-evals' GPQA prompt and extraction; kairyu-bench's fixed choice order.
    spec = SOURCE_BY_NAME["gpqa-diamond"]
    rows = loader.dataset(spec.dataset or "", config="gpqa_diamond", split="train", revision=spec.revision)
    loaded: dict[str, Row] = {}
    for index, row in enumerate(rows):
        correct = str(row["Correct Answer"]).strip()
        incorrect = [str(row[f"Incorrect Answer {n}"]).strip() for n in range(1, 4)]
        choices, expected = deterministic_gpqa_choices(f"gpqa-diamond-{index:04d}", correct, incorrect)
        prompt = SIMPLE_EVALS_MULTICHOICE.format(
            Question=str(row["Question"]).strip(), **dict(zip("ABCD", choices, strict=True))
        )
        loaded[f"gpqa-diamond:{index:04d}"] = {
            "prompt": prompt,
            "expected": expected,
            "domain": row["High-level domain"],
        }
    return loaded


def _gpqa_extract(text: str) -> str | None:
    match = re.search(SIMPLE_EVALS_ANSWER, text)
    return match.group(1) if match else None


_register("gpqa-diamond", _gpqa, lambda row: str(row["domain"]), _gpqa_extract)


# TIGER-AI-Lab/MMLU-Pro f418b116db00b065c2aea046518d8fcf74d39872: the
# evaluate_from_api.py prompt with no exemplars (zero-shot) and the
# evaluate_from_apiX.py extraction (last match per pattern; none is wrong).
MMLU_PRO_HEADER = (
    "The following are multiple choice questions (with answers) about {}. Think step by"
    ' step and then output the answer in the format of "The answer is (X)" at the end.\n\n'
)
MMLU_PRO_PATTERNS = (r"answer is \(?([A-J])\)?", r"[aA]nswer:\s*([A-J])", r"\b[A-J]\b")


def _mmlu_pro_example(question: str, options: list[str]) -> str:
    example = "Question: {}\nOptions: ".format(question)
    for letter, option in zip("ABCDEFGHIJ", options):
        example += "{}. {}\n".format(letter, option)
    return example + "Answer: " + "Let's think step by step." + "\n\n"


def _mmlu_pro(loader: Loader) -> dict[str, Row]:
    spec = SOURCE_BY_NAME["mmlu-pro"]
    rows = loader.dataset(spec.dataset or "", split="test", revision=spec.revision)
    loaded: dict[str, Row] = {}
    for row in rows:
        options = [option for option in row["options"] if option != "N/A"]
        loaded[f"mmlu-pro:{row['question_id']}"] = {
            "prompt": MMLU_PRO_HEADER.format(row["category"]) + _mmlu_pro_example(row["question"], options),
            "expected": row["answer"],
            "category": row["category"],
            "question_id": row["question_id"],
        }
    return loaded


def _mmlu_pro_extract(text: str) -> str | None:
    text = text.replace("**", "")
    for pattern in MMLU_PRO_PATTERNS:
        if matches := re.findall(pattern, text):
            return str(matches[-1])
    return None


# MMLU-ProX translates MMLU-Pro and keeps its question ids.
_register(
    "mmlu-pro", _mmlu_pro, lambda row: str(row["category"]), _mmlu_pro_extract,
    group=lambda row: f"mmlu:{row['question_id']}",
)


# EleutherAI/lm-evaluation-harness d6de81643928d653435c431bae19945d41d32520
# tasks/mmlu_prox (ja): description + doc_to_text at num_fewshot 0, regex
# filter (first match) and exact match ignoring case and punctuation.
MMLU_PROX_JA_SUBJECTS = {
    "biology": "生物学",
    "business": "ビジネス",
    "chemistry": "化学",
    "computer science": "コンピュータサイエンス",
    "economics": "経済学",
    "engineering": "工学",
    "health": "健康科学",
    "history": "歴史",
    "law": "法律",
    "math": "数学",
    "other": "その他",
    "philosophy": "哲学",
    "physics": "物理学",
    "psychology": "心理学",
}
MMLU_PROX_JA_DESCRIPTION = (
    "以下は{subject}に関する選択問題（解答付き）です。段階的に考え、最後に「答えは (X) です」と回答を締めくくってください。"
    "Xは正解の選択肢を示す文字です。\n"
)
MMLU_PROX_JA_PATTERN = r"答えは \(?([ABCDEFGHIJ])\)? です"


def _mmlu_prox_ja(loader: Loader) -> dict[str, Row]:
    spec = SOURCE_BY_NAME["mmlu-prox-ja"]
    rows = loader.dataset(spec.dataset or "", config="ja", split="test", revision=spec.revision)
    loaded: dict[str, Row] = {}
    for row in rows:
        text = "質問：\n" + row["question"] + "\n" + "選択肢：\n"
        for index, letter in enumerate("ABCDEFGHIJ"):
            option = row[f"option_{index}"]
            if option is not None:
                text += "{}. {}\n".format(letter, option)
        text += "回答：一歩一歩考えていきましょう。"
        loaded[f"mmlu-prox-ja:{row['question_id']}"] = {
            "prompt": MMLU_PROX_JA_DESCRIPTION.format(subject=MMLU_PROX_JA_SUBJECTS[row["category"]]) + text,
            "expected": _strip_punctuation(row["answer"]),
            "category": row["category"],
            "question_id": row["question_id"],
        }
    return loaded


def _strip_punctuation(text: str) -> str:
    return text.translate(str.maketrans("", "", string.punctuation)).strip().lower()


def _mmlu_prox_extract(text: str) -> str:
    matches = re.findall(MMLU_PROX_JA_PATTERN, text)
    return _strip_punctuation(matches[0]) if matches else "[invalid]"


_register(
    "mmlu-prox-ja", _mmlu_prox_ja, lambda row: str(row["category"]), _mmlu_prox_extract,
    group=lambda row: f"mmlu:{row['question_id']}",
)


# sbintuitions/flexeval 03f3c56fd7b2f9d4dc044fc0e090c3f8cad73930
# preset_configs/EvalSetup/ja_chat/jamcqa_chat.jsonnet (the dataset author's
# zero-shot chat setup): last `答え: (X)` must equal the gold letter.
JAMC_QA_TEMPLATE = (
    "以下の質問に回答してください。最後に「答え: (E)」のように、「答え: (ここにどの選択肢を選んだか書く)」というフォーマットで回答してください。\n"
    "\n"
    "{question}\n"
    "\n"
    "(A) {choice0}\n"
    "(B) {choice1}\n"
    "(C) {choice2}\n"
    "(D) {choice3}"
)


def _jamc_qa(loader: Loader) -> dict[str, Row]:
    spec = SOURCE_BY_NAME["jamc-qa"]
    rows = loader.dataset(spec.dataset or "", config="v1.0", split="test", revision=spec.revision)
    return {
        f"jamc-qa:{row['qid']}": {
            "prompt": JAMC_QA_TEMPLATE.format(**row),
            "expected": "ABCD"[int(row["answer_index"])],
            "category": row["category"],
        }
        for row in rows
    }


def _jamc_qa_extract(text: str) -> str:
    found = re.findall(r"答え: \((.)\)", text, flags=re.DOTALL)
    return found[-1] if found else ""


_register("jamc-qa", _jamc_qa, lambda row: str(row["category"]), _jamc_qa_extract)
