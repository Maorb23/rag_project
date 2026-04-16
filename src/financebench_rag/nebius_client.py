from __future__ import annotations

import time
from typing import Sequence

from openai import OpenAI

from .config import PipelineConfig


class NebiusChatClient:
    def __init__(self, config: PipelineConfig, timeout: int = 120):
        config.validate_required_secrets()
        self._client = OpenAI(
            base_url=config.api_base_url,
            api_key=config.api_key,
            timeout=timeout,
        )

    def chat(
        self,
        model: str,
        messages: Sequence[dict[str, str]],
        temperature: float = 0.0,
        max_retries: int = 3,
    ) -> str:
        if not model:
            raise ValueError("Model name is empty. Configure model in environment variables.")

        last_error: Exception | None = None
        for attempt in range(1, max_retries + 1):
            try:
                response = self._client.chat.completions.create(
                    model=model,
                    messages=list(messages),
                    temperature=temperature,
                )
                return response.choices[0].message.content or ""
            except Exception as exc:  # pragma: no cover
                last_error = exc
                if attempt < max_retries:
                    time.sleep(2 ** attempt)

        raise RuntimeError(f"Nebius chat call failed after {max_retries} attempts: {last_error}")
