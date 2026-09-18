"""
Thin wrapper around the Groq chat-completions API.

Kept separate from prompt.py and parser.py so the LLM call itself
(network, retries, timeouts) can be mocked out independently in tests.
"""

from groq import Groq

from src import config

_client: Groq | None = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=config.GROQ_API_KEY)
    return _client


def call_llm(messages: list[dict], max_retries: int = 2) -> str:
    """
    Call Groq's chat completion endpoint and return the raw text response.
    Retries on transient failures. Raises the last exception if all
    attempts fail so the caller can surface a clean error to the user.
    """
    client = _get_client()
    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=config.GROQ_MODEL,
                messages=messages,
                temperature=0,
                max_tokens=500,
                response_format={"type": "json_object"},
            )
            return response.choices[0].message.content
        except Exception as exc:  # noqa: BLE001 - we want to retry on any transient error
            last_error = exc

    raise RuntimeError(f"Groq API call failed after {max_retries + 1} attempts: {last_error}")
