import json
import os
import re
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass(slots=True)
class LLMResponse:
    model: str
    raw_text: str
    data: dict


class LLMError(RuntimeError):
    """Base error for local model availability and generation failures."""


class LLMUnavailableError(LLMError):
    pass


class LLMGenerationError(LLMError):
    pass


class OllamaClient:
    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        timeout: float = 120.0,
    ):
        self.model = model or os.getenv("OLLAMA_MODEL", "qwen3:4b")
        self.base_url = (
            base_url or os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
        ).rstrip("/")
        self.timeout = timeout
        self.last_error: str | None = None

    def available(self) -> bool:
        request = Request(f"{self.base_url}/api/tags", method="GET")

        try:
            with urlopen(request, timeout=3) as response:
                if response.status != 200:
                    self.last_error = f"Ollama returned HTTP {response.status}."
                    return False
                payload = json.loads(response.read().decode("utf-8"))
                installed = {
                    item.get("name")
                    for item in payload.get("models", [])
                    if isinstance(item, dict)
                }
                if self.model not in installed:
                    self.last_error = (
                        f"Configured model {self.model!r} is not installed. "
                        f"Installed models: {sorted(item for item in installed if item)}"
                    )
                    return False
                self.last_error = None
                return True
        except HTTPError as error:
            self.last_error = f"Ollama availability HTTP error {error.code}: {error.reason}"
            return False
        except (OSError, URLError, ValueError, json.JSONDecodeError) as error:
            self.last_error = f"Ollama availability error: {error}"
            return False

    def generate_json(self, prompt: str) -> LLMResponse:
        if not self.available():
            raise LLMUnavailableError(self.last_error or "Ollama model unavailable.")

        payload = json.dumps({
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0},
        }).encode("utf-8")

        request = Request(
            f"{self.base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            if "context" in detail.lower():
                raise LLMGenerationError(
                    f"Ollama context-length error: {detail}"
                ) from error
            raise LLMGenerationError(
                f"Ollama HTTP error {error.code}: {detail or error.reason}"
            ) from error
        except TimeoutError as error:
            raise LLMGenerationError("Ollama request timed out.") from error
        except (OSError, URLError, ValueError, json.JSONDecodeError) as error:
            raise LLMGenerationError(f"Ollama response error: {error}") from error

        if not isinstance(body, dict):
            raise LLMGenerationError("Ollama response body must be an object.")
        if body.get("error"):
            message = str(body["error"])
            if "context" in message.lower():
                message = f"context-length error: {message}"
            raise LLMGenerationError(message)
        if not isinstance(body.get("response"), str):
            raise LLMGenerationError(
                "Ollama response is missing its string 'response' field."
            )

        raw_text = body.get("response", "")
        try:
            data = parse_json_object(raw_text)
        except (ValueError, json.JSONDecodeError) as error:
            raise LLMGenerationError(
                f"Ollama returned malformed JSON: {error}"
            ) from error

        return LLMResponse(
            model=self.model,
            raw_text=raw_text,
            data=data,
        )


def parse_json_object(text: str) -> dict:
    candidate = text.strip()
    candidate = re.sub(r"^```(?:json)?\s*", "", candidate)
    candidate = re.sub(r"\s*```$", "", candidate)
    data = json.loads(candidate)

    if not isinstance(data, dict):
        raise ValueError("The LLM response must be a JSON object.")

    return data