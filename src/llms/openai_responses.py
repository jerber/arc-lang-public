import asyncio
import json
import time
import typing as T

from openai import AsyncOpenAI
from openai.types.responses.response import Response

from src.llms.models import Model
from src.log import log


RESPONSES_EXTRA_HEADERS = {"OpenAI-Beta": "responses=v2"}
POLL_TERMINAL_STATUSES = {
    "completed",
    "cancelled",
    "failed",
    "expired",
    "succeeded",
}
POLL_ERROR_STATUSES = {
    "cancelled",
    "failed",
    "expired",
    "rejected",
}
POLL_DEFAULT_INTERVAL = 2.0
POLL_MAX_INTERVAL = 15.0
POLL_TIMEOUT_SECONDS = 10_800.0

OPENAI_MODEL_MAX_OUTPUT_TOKENS: dict[Model, int] = {
    Model.gpt_4_5: 100_000,
    Model.gpt_4_o: 100_000,
    Model.o3_mini: 100_000,
    Model.o3_mini_high: 100_000,
    Model.o4_mini_high: 128_000,
    Model.o3: 128_000,
    Model.o3_pro: 128_000,
    Model.o4_mini: 128_000,
    Model.gpt_4_1: 128_000,
    Model.gpt_4_1_mini: 128_000,
    Model.gpt_5: 128_000,
    Model.gpt_52: 128_000,
    Model.gpt_5_pro: 200_000,
}


async def create_and_poll_response(
    client: AsyncOpenAI,
    *,
    model: Model,
    create_kwargs: dict[str, T.Any],
) -> Response:
    """
    Create and poll OpenAI Response API requests with background mode support.

    For GPT-5 models:
    - Uses background mode for long-running tasks (no streaming)
    - Sets verbosity to 'low' for GPT-5-Pro to reduce token usage
    - Enforces store=True (required for background mode)
    """
    create_kwargs = dict(create_kwargs)
    extra_body: dict[str, T.Any] = dict(create_kwargs.pop("extra_body", {}) or {})
    extra_headers: dict[str, T.Any] = dict(create_kwargs.pop("extra_headers", {}) or {})
    headers = {**RESPONSES_EXTRA_HEADERS, **extra_headers}

    is_gpt5 = model in {Model.gpt_5, Model.gpt_52, Model.gpt_5_pro}

    # Configure GPT-5 defaults: background mode with polling
    if is_gpt5:
        create_kwargs["store"] = True
        extra_body.setdefault("background", True)

        # Set low verbosity for GPT-5-Pro to reduce token usage
        if model == Model.gpt_5_pro:
            text_config = create_kwargs.setdefault("text", {})
            if isinstance(text_config, dict):
                text_config.setdefault("verbosity", "low")
    else:
        create_kwargs.setdefault("store", True)

    if extra_body:
        create_kwargs["extra_body"] = extra_body

    log.info(
        "openai_request_config",
        model=model.value,
        background=extra_body.get("background", False),
        store=create_kwargs.get("store"),
        verbosity=create_kwargs.get("text", {}).get("verbosity") if isinstance(create_kwargs.get("text"), dict) else None,
        tools=len(create_kwargs.get("tools", [])),
    )

    return await _handle_polling_response(
        client=client,
        model=model,
        create_kwargs=create_kwargs,
        headers=headers,
    )


async def _handle_polling_response(
    client: AsyncOpenAI,
    model: Model,
    create_kwargs: dict[str, T.Any],
    headers: dict[str, T.Any],
) -> Response:
    """Create and poll response until completion with retry logic for server errors."""
    max_retries = 3
    retry_delay = 5.0

    for attempt in range(max_retries):
        try:
            resp = await client.responses.create(extra_headers=headers, **create_kwargs)

            log.info(
                "openai_response_created",
                model=model.value,
                response_id=resp.id,
                status=resp.status,
                attempt=attempt + 1,
            )

            poll_interval = POLL_DEFAULT_INTERVAL
            start_time = time.time()

            # Poll while queued or in_progress
            while resp.status in {"queued", "in_progress"}:
                if time.time() - start_time > POLL_TIMEOUT_SECONDS:
                    raise TimeoutError(f"Response polling timeout after {POLL_TIMEOUT_SECONDS}s for {model.value}")

                log.info(
                    "openai_response_polling",
                    model=model.value,
                    response_id=resp.id,
                    status=resp.status,
                )

                await asyncio.sleep(poll_interval)
                poll_interval = min(poll_interval * 1.5, POLL_MAX_INTERVAL)
                resp = await client.responses.retrieve(resp.id, extra_headers=headers)

            # Check final status
            log.info(
                "openai_response_final",
                model=model.value,
                response_id=resp.id,
                status=resp.status,
            )

            # Check for transient errors that should be retried
            if resp.status == "failed" and resp.error:
                error_code = resp.error.code if resp.error else None
                # Retry server errors and rate limits
                if error_code in {"server_error", "rate_limit_exceeded"} and attempt < max_retries - 1:
                    log.warning(
                        "openai_transient_error_retry",
                        model=model.value,
                        response_id=resp.id,
                        error_code=error_code,
                        attempt=attempt + 1,
                        max_retries=max_retries,
                        error_message=resp.error.message if resp.error else None,
                    )
                    await asyncio.sleep(retry_delay * (attempt + 1))
                    continue  # Retry with new request

            # Non-retryable errors
            if resp.status in POLL_ERROR_STATUSES:
                raise RuntimeError(f"Response {resp.status} for {model.value}: {resp.error or resp.model_dump()}")
            if resp.status == "requires_action":
                raise RuntimeError(f"Response requires action (not supported) for {model.value}")

            return resp

        except TimeoutError:
            # Don't retry timeouts
            raise
        except Exception as e:
            if attempt < max_retries - 1:
                log.warning(
                    "openai_request_error_retry",
                    model=model.value,
                    attempt=attempt + 1,
                    max_retries=max_retries,
                    error=str(e),
                )
                await asyncio.sleep(retry_delay * (attempt + 1))
            else:
                raise

    raise RuntimeError(f"Max retries exceeded for {model.value}")


def extract_structured_output(response: Response | dict[str, T.Any]) -> dict[str, T.Any]:
    """Extract structured JSON from Response API (json_schema format)."""
    payload = response.model_dump() if isinstance(response, Response) else response

    # Primary: Direct JSON field in content (json_schema format)
    for item in payload.get("output", []):
        for content in item.get("content") or []:
            if json_data := content.get("json"):
                return json_data

    # Fallback: Parse text as JSON
    for item in payload.get("output", []):
        for content in item.get("content") or []:
            if text := content.get("text"):
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    continue

    # Last resort: Parse output_text property
    if output_text := payload.get("output_text"):
        try:
            return json.loads(output_text)
        except json.JSONDecodeError:
            pass

    raise ValueError("No structured JSON found in response")
