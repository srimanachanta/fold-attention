"""LongBench v1's sixteen English tasks as the benchmark defines them.

Prompts, generation lengths, the tasks answered without a chat template and
the metrics are copied from THUDM/LongBench (`LongBench/config/*.json`,
`metrics.py`, `eval.py`, `pred.py`), so scores compare with published ones.
The data is the benchmark's `data.zip` from the Hugging Face hub.
"""

from __future__ import annotations

import json
import re
import string
import zipfile
from collections import Counter

TASKS = (
    "narrativeqa",
    "qasper",
    "multifieldqa_en",
    "hotpotqa",
    "2wikimqa",
    "musique",
    "gov_report",
    "qmsum",
    "multi_news",
    "trec",
    "triviaqa",
    "samsum",
    "passage_count",
    "passage_retrieval_en",
    "lcc",
    "repobench-p",
)

CATEGORY = {
    "narrativeqa": "single-doc QA",
    "qasper": "single-doc QA",
    "multifieldqa_en": "single-doc QA",
    "hotpotqa": "multi-doc QA",
    "2wikimqa": "multi-doc QA",
    "musique": "multi-doc QA",
    "gov_report": "summarization",
    "qmsum": "summarization",
    "multi_news": "summarization",
    "trec": "few-shot",
    "triviaqa": "few-shot",
    "samsum": "few-shot",
    "passage_count": "synthetic",
    "passage_retrieval_en": "synthetic",
    "lcc": "code",
    "repobench-p": "code",
}

MAXLEN = {
    "narrativeqa": 128,
    "qasper": 128,
    "multifieldqa_en": 64,
    "hotpotqa": 32,
    "2wikimqa": 32,
    "musique": 32,
    "gov_report": 512,
    "qmsum": 512,
    "multi_news": 512,
    "trec": 64,
    "triviaqa": 32,
    "samsum": 128,
    "passage_count": 32,
    "passage_retrieval_en": 32,
    "lcc": 64,
    "repobench-p": 64,
}

# pred.py leaves these without a chat template
NO_CHAT = {"trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"}
# eval.py scores only the first line of these
FIRST_LINE = {"trec", "triviaqa", "samsum", "lsht"}

_QA = (
    "Answer the question based on the given passages. Only give me the answer and do not "
    "output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the "
    "question based on the given passages. Only give me the answer and do not output any "
    "other words.\n\nQuestion: {input}\nAnswer:"
)
_QASPER_RULE = (
    "Answer the question as concisely as you can, using a single phrase or sentence if "
    "possible. If the question cannot be answered based on the information in the article, "
    'write "unanswerable". If the question is a yes/no question, answer "yes", "no", or '
    '"unanswerable". Do not provide any explanation.'
)
# "asconcisely" is the benchmark's own spelling
PROMPT = {
    "narrativeqa": (
        "You are given a story, which can be either a novel or a movie script, and a question. "
        "Answer the question asconcisely as you can, using a single phrase if possible. Do not "
        "provide any explanation.\n\nStory: {context}\n\nNow, answer the question based on the "
        "story asconcisely as you can, using a single phrase if possible. Do not provide any "
        "explanation.\n\nQuestion: {input}\n\nAnswer:"
    ),
    "qasper": (
        "You are given a scientific article and a question. "
        + _QASPER_RULE
        + "\n\nArticle: {context}\n\n Answer the question based on the above article as "
        "concisely as you can, using a single phrase or sentence if possible. If the question "
        'cannot be answered based on the information in the article, write "unanswerable". If '
        'the question is a yes/no question, answer "yes", "no", or "unanswerable". Do not '
        "provide any explanation.\n\nQuestion: {input}\n\nAnswer:"
    ),
    "multifieldqa_en": (
        "Read the following text and answer briefly.\n\n{context}\n\nNow, answer the following "
        "question based on the above text, only give me the answer and do not output any other "
        "words.\n\nQuestion: {input}\nAnswer:"
    ),
    "hotpotqa": _QA,
    "2wikimqa": _QA,
    "musique": _QA,
    "gov_report": (
        "You are given a report by a government agency. Write a one-page summary of the "
        "report.\n\nReport:\n{context}\n\nNow, write a one-page summary of the report.\n\n"
        "Summary:"
    ),
    "qmsum": (
        "You are given a meeting transcript and a query containing a question or instruction. "
        "Answer the query in one or more sentences.\n\nTranscript:\n{context}\n\nNow, answer the "
        "query based on the above meeting transcript in one or more sentences.\n\nQuery: "
        "{input}\nAnswer:"
    ),
    "multi_news": (
        "You are given several news passages. Write a one-page summary of all news. \n\n"
        "News:\n{context}\n\nNow, write a one-page summary of all the news.\n\nSummary:"
    ),
    "trec": (
        "Please determine the type of the question below. Here are some examples of "
        "questions.\n\n{context}\n{input}"
    ),
    "triviaqa": (
        "Answer the question based on the given passage. Only give me the answer and do not "
        "output any other words. The following are some examples.\n\n{context}\n\n{input}"
    ),
    "samsum": (
        "Summarize the dialogue into a few short sentences. The following are some examples."
        "\n\n{context}\n\n{input}"
    ),
    "passage_count": (
        "There are some paragraphs below sourced from Wikipedia. Some of them may be "
        "duplicates. Please carefully read these paragraphs and determine how many unique "
        "paragraphs there are after removing duplicates. In other words, how many "
        "non-repeating paragraphs are there in total?\n\n{context}\n\nPlease enter the final "
        "count of unique paragraphs after removing duplicates. The output format should only "
        "contain the number, such as 1, 2, 3, and so on.\n\nThe final answer is: "
    ),
    "passage_retrieval_en": (
        "Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine which "
        "paragraph the abstract is from.\n\n{context}\n\nThe following is an abstract.\n\n"
        "{input}\n\nPlease enter the number of the paragraph that the abstract is from. The "
        'answer format must be like "Paragraph 1", "Paragraph 2", etc.\n\nThe answer is: '
    ),
    "lcc": "Please complete the code given below. \n{context}Next line of code:\n",
    "repobench-p": "Please complete the code given below. \n{context}{input}Next line of code:\n",
}


def load(task: str) -> list[dict]:
    """The task's test samples from the benchmark's `data.zip`."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download("THUDM/LongBench", "data.zip", repo_type="dataset")
    with zipfile.ZipFile(path) as z, z.open(f"data/{task}.jsonl") as f:
        return [json.loads(line) for line in f]


# ---------------------------------------------------------------- metrics


def normalize_answer(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def _f1(pred, truth):
    common = Counter(pred) & Counter(truth)
    same = sum(common.values())
    if same == 0:
        return 0.0
    p, r = same / len(pred), same / len(truth)
    return 2 * p * r / (p + r)


def qa_f1_score(prediction, ground_truth, **kw):
    return _f1(normalize_answer(prediction).split(), normalize_answer(ground_truth).split())


def rouge_score(prediction, ground_truth, **kw):
    from rouge import Rouge

    try:
        return Rouge().get_scores([prediction], [ground_truth], avg=True)["rouge-l"]["f"]
    except Exception:  # noqa: BLE001
        # the benchmark scores what Rouge rejects (an empty prediction) as 0
        return 0.0


def classification_score(prediction, ground_truth, **kw):
    matched = [c for c in kw["all_classes"] if c in prediction]
    for m in list(matched):
        if m in ground_truth and m != ground_truth:
            matched.remove(m)
    return 1.0 / len(matched) if ground_truth in matched else 0.0


def _share(numbers, want):
    return 0.0 if not numbers else sum(str(n) == str(want) for n in numbers) / len(numbers)


def count_score(prediction, ground_truth, **kw):
    return _share(re.findall(r"\d+", prediction), ground_truth)


def retrieval_score(prediction, ground_truth, **kw):
    want = re.findall(r"Paragraph (\d+)", ground_truth)[0]
    return _share(re.findall(r"\d+", prediction), want)


def code_sim_score(prediction, ground_truth, **kw):
    from fuzzywuzzy import fuzz

    line = ""
    for x in prediction.lstrip("\n").split("\n"):
        if "`" not in x and "#" not in x and "//" not in x:
            line = x
            break
    return fuzz.ratio(line, ground_truth) / 100


METRIC = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "passage_count": count_score,
    "passage_retrieval_en": retrieval_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}


def score(task: str, prediction: str, sample: dict) -> float:
    """One prediction's score, `eval.py`'s best over the reference answers."""
    if task in FIRST_LINE:
        prediction = prediction.lstrip("\n").split("\n")[0]
    return max(
        METRIC[task](prediction, gt, all_classes=sample.get("all_classes"))
        for gt in sample["answers"]
    )
