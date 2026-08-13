from __future__ import annotations

import json
import os
from typing import Any, Literal

from onebee.evaluation.graders.judge import JudgeVerdict

# NOTE: the `openai` package is imported lazily inside method bodies. It is NOT a
# hard dependency of the package — it lives in the optional `judge` extra — so this
# module (and everything else in `onebee`) must import cleanly without it installed.


class OpenAIJudge:
    """A :class:`Judge` backed by the OpenAI chat-completions API in JSON mode.

    The ``openai`` client is constructed per-request so that construction of this
    object never requires the package (or an API key) to be present.
    """

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.0,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.temperature = temperature

    def _resolve_api_key(self) -> str | None:
        if self.api_key is not None:
            return self.api_key
        return os.environ.get("OPENAI_API_KEY")

    def _client(self) -> Any:
        import openai

        kwargs: dict[str, Any] = {}
        api_key = self._resolve_api_key()
        if api_key is not None:
            kwargs["api_key"] = api_key
        if self.base_url is not None:
            kwargs["base_url"] = self.base_url
        return openai.OpenAI(**kwargs)

    def _request_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        client = self._client()
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        raw: str | None = None
        for attempt in range(2):
            completion = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                response_format={"type": "json_object"},
            )
            raw = completion.choices[0].message.content

            try:
                parsed = json.loads(raw) if raw else None
            except json.JSONDecodeError:
                parsed = None

            if isinstance(parsed, dict):
                return parsed

            messages.append({"role": "assistant", "content": raw or ""})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Your previous response was not valid JSON. "
                        "Return ONLY a single valid JSON object and nothing else."
                    ),
                }
            )

        # TODO: no exponential backoff / rate-limit handling here (out of scope for
        # the bake-off); a production judge would retry on 429/5xx with backoff.
        raise RuntimeError(
            "OpenAIJudge: judge model returned an unparseable response even after a "
            f"retry. Raw response: {raw!r}"
        )

    def score_response(self, question: str, response: str, rubric: str) -> JudgeVerdict:
        system_prompt = (
            "You are a rigorous evaluation judge. Score the response against the "
            "rubric on a scale of 0.0 to 5.0. You MUST respond with a single JSON "
            'object containing exactly two keys: "score" (a float from 0.0 to 5.0) '
            'and "justification" (a string explaining the score). Output valid JSON '
            "only, with no other text."
        )
        user_prompt = (
            f"Question:\n{question}\n\nResponse:\n{response}\n\nRubric:\n{rubric}\n\n"
            "Score the response on a scale of 0 to 5 against the rubric."
        )
        data = self._request_json(system_prompt, user_prompt)
        try:
            score = float(data["score"])
            justification = str(data["justification"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"OpenAIJudge: malformed verdict JSON from judge: {data!r}") from exc
        return JudgeVerdict(score=score, justification=justification)

    def compare_pairwise(
        self,
        question: str,
        response_a: str,
        response_b: str,
        rubric: str,
        order: Literal["AB", "BA"],
    ) -> JudgeVerdict:
        system_prompt = (
            "You are a rigorous evaluation judge comparing two responses to a "
            "question. You MUST respond with a single JSON object containing exactly "
            'two keys: "score" (one of 0.0, 0.5, or 1.0, where 0.0 means response A '
            "is better, 1.0 means response B is better, and 0.5 means a tie) and "
            '"justification" (a string). Output valid JSON only, with no other text.'
        )
        user_prompt = (
            f"Question:\n{question}\n\nResponse A:\n{response_a}\n\n"
            f"Response B:\n{response_b}\n\nRubric:\n{rubric}\n\n"
            "Return 0.0 if A is better, 1.0 if B is better, 0.5 if tied."
        )
        data = self._request_json(system_prompt, user_prompt)
        try:
            score = float(data["score"])
            justification = str(data["justification"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"OpenAIJudge: malformed verdict JSON from judge: {data!r}") from exc
        return JudgeVerdict(score=score, justification=justification, order=order)
