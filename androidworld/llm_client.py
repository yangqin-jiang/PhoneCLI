"""OpenAI-compatible API client for text LLM and vision (VLM) calls."""

import base64
import sys
from typing import Optional

from aworld_agent.token_usage import token_usage


def _get_client(api_key: str, api_base: str):
    """Lazy import OpenAI client."""
    from openai import OpenAI
    return OpenAI(api_key=api_key, base_url=api_base, timeout=60.0)


def _encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _completion_extra_kwargs(api_base: str, model: str) -> dict:
    """Provider-specific options shared by text and vision completions."""
    if "deepseek" in api_base.lower() or "deepseek" in model.lower():
        return {"extra_body": {"thinking": {"type": "disabled"}}}
    if (
        "openrouter.ai" in api_base.lower()
        and model.lower() == "qwen/qwen3.7-plus"
    ):
        # Strict classifier output and one-line UI actions both require a final
        # answer.  With OpenRouter's default reasoning this model can spend the
        # completion budget on reasoning and return message.content=None.
        return {"extra_body": {"reasoning": {"effort": "none"}}}
    return {}


def _record_usage(response, label: str = ""):
    """Extract usage from OpenAI/OpenRouter response and accumulate."""
    try:
        usage = response.usage
        token_usage.add(
            prompt_tokens=getattr(usage, 'prompt_tokens', 0) or 0,
            completion_tokens=getattr(usage, 'completion_tokens', 0) or 0,
            cache_read_tokens=getattr(usage, 'cache_read_input_tokens', 0) or 0,
            cache_write_tokens=getattr(usage, 'cache_creation_input_tokens', 0) or 0,
            label=label,
        )
    except Exception:
        pass


def text_completion(
    system_prompt: str,
    user_prompt: str,
    api_key: str = "EMPTY",
    api_base: str = "http://localhost:8002/v1",
    model: str = "Qwen/Qwen2.5-3B-Instruct",
    max_tokens: int = 4096,
    temperature: float = 0.0,
    label: str = "text",
) -> str:
    """Send a text-only completion request."""
    client = _get_client(api_key, api_base)
    extra = _completion_extra_kwargs(api_base, model)

    try:
        r = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            **extra,
        )
        content = r.choices[0].message.content or ""
        if not content and hasattr(r.choices[0].message, "reasoning_content"):
            content = r.choices[0].message.reasoning_content or ""
        _record_usage(r, label=label)
        return content
    except Exception as e:
        print(f"LLM error: {e}", file=sys.stderr)
        raise


def vision_completion(
    system_prompt: str,
    user_text: str,
    image_paths: list[str],
    api_key: str = "EMPTY",
    api_base: str = "http://localhost:8002/v1",
    model: str = "Qwen/Qwen2.5-3B-Instruct",
    max_tokens: int = 4096,
    temperature: float = 0.0,
    label: str = "vision",
) -> str:
    """Send a vision (image + text) completion request."""
    client = _get_client(api_key, api_base)
    extra = _completion_extra_kwargs(api_base, model)

    content = [{"type": "text", "text": user_text}]
    for path in image_paths:
        b64 = _encode_image(path)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"},
        })

    try:
        for attempt in range(2):
            r = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                **extra,
            )
            _record_usage(r, label=label)
            answer = r.choices[0].message.content
            if isinstance(answer, str) and answer.strip():
                return answer
            if attempt == 0:
                print(
                    "VLM warning: successful response had no final content; "
                    "retrying",
                    file=sys.stderr,
                )
        raise RuntimeError("VLM returned no final content after 2 attempts")
    except Exception as e:
        print(f"VLM error: {e}", file=sys.stderr)
        raise
