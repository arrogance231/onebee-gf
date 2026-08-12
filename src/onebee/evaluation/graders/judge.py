from __future__ import annotations

from typing import Literal, Protocol

from pydantic import BaseModel


class JudgeVerdict(BaseModel):
    score: float
    justification: str
    order: Literal["AB", "BA"] | None = None


class Judge(Protocol):
    def score_response(self, question: str, response: str, rubric: str) -> JudgeVerdict:
        ...

    def compare_pairwise(
        self,
        question: str,
        response_a: str,
        response_b: str,
        rubric: str,
        order: Literal["AB", "BA"],
    ) -> JudgeVerdict:
        ...


class FakeJudge:
    def score_response(self, question: str, response: str, rubric: str) -> JudgeVerdict:
        word_count = len(response.split()) if response.strip() else 0
        score = min(5.0, word_count / 10)
        return JudgeVerdict(
            score=score,
            justification=(
                f"Deterministic score from {word_count} words: "
                f"min(5.0, {word_count}/10) = {score:.2f}"
            ),
        )

    def compare_pairwise(
        self,
        question: str,
        response_a: str,
        response_b: str,
        rubric: str,
        order: Literal["AB", "BA"],
    ) -> JudgeVerdict:
        len_a = len(response_a.split()) if response_a.strip() else 0
        len_b = len(response_b.split()) if response_b.strip() else 0
        if len_a > len_b:
            score = 1.0
            justification = f"Response A ({len_a} words) is longer than B ({len_b} words)"
        elif len_b > len_a:
            score = 0.0
            justification = f"Response B ({len_b} words) is longer than A ({len_a} words)"
        else:
            score = 0.5
            justification = (
                f"Both responses have {len_a} words; tie broken to 0.5"
            )
        return JudgeVerdict(score=score, justification=justification, order=order)


def dual_order_score(
    judge: Judge,
    question: str,
    response_a: str,
    response_b: str,
    rubric: str,
) -> float:
    verdict_ab = judge.compare_pairwise(question, response_a, response_b, rubric, "AB")
    verdict_ba = judge.compare_pairwise(question, response_b, response_a, rubric, "BA")
    return (verdict_ab.score + (1.0 - verdict_ba.score)) / 2.0
