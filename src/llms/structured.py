import asyncio
import copy
import functools
import json
import os
import random
import time
import typing as T

from anthropic import AsyncAnthropic
from devtools import debug
from google import genai
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings
import httpx
from pydantic_ai.providers.gateway import gateway_provider
from xai_sdk import AsyncClient as XaiAsyncClient
from xai_sdk.chat import assistant, image, system, user

from src.async_utils.semaphore_monitor import MonitoredSemaphore
from src.llms.models import Model
from src.llms.openai_responses import (
    OPENAI_MODEL_MAX_OUTPUT_TOKENS,
    create_and_poll_response,
    extract_structured_output,
)
from src.log import log
from src.utils import random_str

BMType = T.TypeVar("BMType", bound=BaseModel)


P = T.ParamSpec("P")
R = T.TypeVar("R")


def retry_with_backoff(
    max_retries: int,
    base_delay: float = 3,
    max_delay: float = 120,
) -> T.Callable[[T.Callable[P, T.Awaitable[R]]], T.Callable[P, T.Awaitable[R]]]:
    """
    Decorator for *async* functions that retries transient “UNAVAILABLE /
    RESOURCE_EXHAUSTED”-style errors with exponential back-off.

    • Executes up to `max_retries + 1` total attempts (first try + N retries).
    • Full-jitter back-off — waits a random time in `[0, base_delay × 2**(n-1)]`.
    • Classifies retryability with simple string matching for readability.
    """

    def decorator(fn: T.Callable[P, T.Awaitable[R]]) -> T.Callable[P, T.Awaitable[R]]:
        @functools.wraps(fn)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            for attempt in range(1, max_retries + 2):  # 1-based
                start = time.time()
                try:
                    res = await fn(*args, **kwargs)
                    if attempt > 1:
                        log.debug(
                            "retry_succeeded!", function=fn.__name__, attempt=attempt
                        )
                    return res
                except asyncio.CancelledError:  # never retry cancellations
                    raise

                except Exception as exc:  # noqa: BLE001
                    duration = time.time() - start
                    msg = str(exc)

                    # ---- simple, readable retry classification ----
                    retryable = (
                        "UNAVAILABLE" in msg.upper()
                        or "RESOURCE_EXHAUSTED" in msg.upper()
                        or "StatusCode.UNAVAILABLE" in msg
                        or "StatusCode.RESOURCE_EXHAUSTED" in msg
                        or "StatusCode.UNKNOWN" in msg
                        or "Empty response from OpenRouter model" in msg
                        or "validation error" in msg
                        or "SAFETY_CHECK_TYPE_BIO" in msg
                        or "524" in msg  # Cloudflare timeout
                        or "timeout" in msg.lower()
                        or "ServerError" in msg
                    )
                    if "StatusCode.DEADLINE_EXCEEDED" in msg:
                        retryable = False
                    if duration > 1_000:
                        retryable = False
                    if duration > 500 and attempt > 2:
                        retryable = False

                    if not retryable or attempt > max_retries:
                        log.error(
                            "retry_failed",
                            function=fn.__name__,
                            attempt=attempt,
                            duration_seconds=duration,
                            error=msg,
                            error_type=type(exc).__name__,
                            max_retries_reached=(attempt > max_retries),
                        )
                        raise

                    # ---- full-jitter exponential back-off ----
                    base_wait = min(base_delay * 2 ** (attempt - 1), max_delay)
                    wait = random.uniform(0, base_wait)

                    log.warn(
                        "retry_attempt",
                        function=fn.__name__,
                        attempt=attempt,
                        duration_seconds=duration,
                        wait_seconds=wait,
                        error=msg,
                        error_type=type(exc).__name__,
                    )
                    await asyncio.sleep(wait)

            # should never reach here
            raise RuntimeError("retry_with_backoff: fell through unexpectedly")

        return wrapper

    return decorator


openai_client = AsyncOpenAI(
    api_key=os.environ["OPENAI_API_KEY"], timeout=10_800, max_retries=2
)
anthropic_client = AsyncAnthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY"), timeout=3_010, max_retries=2
)
deepseek_client = AsyncOpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
    timeout=2500,
    max_retries=2,
)
openrouter_client = AsyncOpenAI(
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
    timeout=2500,
    max_retries=2,
)
gemini_client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"],
)

# Semaphore to limit concurrent API calls to 100
API_SEMAPHORE = MonitoredSemaphore(
    int(os.environ["MAX_CONCURRENCY"]), name="API_SEMAPHORE"
)


async def get_next_structure(
    structure: type[BMType],  # type[T]
    model: Model,
    messages: list,
) -> BMType:
    res_id = random_str(k=6)

    with log.span(
        "llm_call",
        model=model.value,
        structure=structure.__name__,
        request_id=res_id,
    ) as span:
        start = time.time()
        log.debug(
            "Starting LLM call",
            model=model.value,
            structure=structure.__name__,
            request_id=res_id,
        )

        async with API_SEMAPHORE:
            if model in [
                Model.o4_mini,
                Model.o3,
                Model.gpt_4_1,
                Model.gpt_4_1_mini,
                Model.o3_pro,
                Model.gpt_5,
                Model.gpt_52,
                Model.gpt_5_pro,
            ]:
                res = await _get_next_structure_openai(
                    structure=structure, model=model, messages=messages
                )
            elif model in [Model.sonnet_4, Model.opus_4, Model.sonnet_4_5]:
                res = await _get_next_structure_anthropic(
                    structure=structure, model=model, messages=messages
                )
            elif model in [Model.grok_4, Model.grok_3_mini_fast]:
                res = await _get_next_structure_xai(
                    structure=structure, model=model, messages=messages
                )
            elif model in [Model.deepseek_reasoner, Model.deepseek_chat]:
                res = await _get_next_structure_deepseek(
                    structure=structure, model=model, messages=messages
                )
            elif model in [
                Model.gemini_2_5,
                Model.gemini_2_5_flash_lite,
                Model.gemini_3_pro,
            ]:
                res = await _get_next_structure_gemini(
                    structure=structure, model=model, messages=messages
                )
            elif model in [Model.gemini_3_pro_gateway]:
                res = await _get_next_structure_pydantic_gateway(
                    structure=structure, model=model, messages=messages
                )
            elif model in [
                Model.openrouter_sonnet_3_7_thinking,
                Model.openrouter_sonnet_3_7,
                Model.openrouter_gemini_2_5_free,
                Model.openrouter_gemini_2_5,
                Model.openrouter_deepseek_3_free,
                Model.openrouter_deepseek_r1,
                Model.openrouter_deepseek_r1_free,
                Model.openrouter_grok_v3,
                Model.openrouter_quasar_alpha,
                Model.openrouter_optimus_alpha,
                Model.openrouter_qwen_235b,
                Model.openrouter_qwen_235b_thinking,
                Model.openrouter_gemini_2_5_flash_lite,
                Model.openrouter_glm,
                Model.openrouter_kimi_k2,
                Model.openrouter_grok_4,
                Model.openrouter_horizon_alpha,
                Model.openrouter_gpt_oss_120b,
                Model.gemini_3_pro_openrouter,
            ]:
                res = await _get_next_structure_openrouter(
                    structure=structure, model=model, messages=messages
                )
            else:
                raise Exception(f"Invalid model {model}.")

            duration = time.time() - start
            span.set_attribute("duration_seconds", duration)
            # span.set_attribute("response", res.model_dump())

            response_dump = res.model_dump()
            response_keys = list(response_dump.keys())
            if os.getenv("LOG_GRIDS", "0") == "1":
                pass
            else:
                response_dump = {}

            log.debug(
                "LLM call completed",
                model=model.value,
                structure=structure.__name__,
                duration_seconds=duration,
                request_id=res_id,
                response=response_dump,
                response_keys=response_keys,
            )

            return res


async def _get_next_structure_openai(
    structure: type[BMType],  # type[T]
    model: Model,
    messages: list,
) -> BMType:
    reasoning: dict[str, str] | None = None
    if model in [Model.o3, Model.o4_mini, Model.o3_pro, Model.gpt_5, Model.gpt_52, Model.gpt_5_pro]:
        reasoning = {"effort": "high"}

    max_output_tokens = OPENAI_MODEL_MAX_OUTPUT_TOKENS.get(model, 128_000)

    schema = structure.model_json_schema()
    if "additionalProperties" not in schema:
        schema["additionalProperties"] = False

    create_kwargs: dict[str, T.Any] = {
        "model": model.value,
        "input": messages,
        "max_output_tokens": max_output_tokens,
        "text": {
            "format": {
                "type": "json_schema",
                "name": structure.__name__,
                "schema": schema,
                "strict": True,
            }
        },
    }
    if reasoning:
        create_kwargs["reasoning"] = reasoning

    raw_response = await create_and_poll_response(
        openai_client,
        model=model,
        create_kwargs=create_kwargs,
    )

    usage = raw_response.usage or {}
    input_token_details = getattr(usage, "input_token_details", None) or {}
    output_token_details = getattr(usage, "output_tokens_details", None) or {}

    openai_usage = OpenAIUsage(
        completion_tokens=int(getattr(usage, "output_tokens", 0) or 0),
        prompt_tokens=int(getattr(usage, "input_tokens", 0) or 0),
        total_tokens=int(getattr(usage, "total_tokens", 0) or 0),
        reasoning_tokens=int(getattr(output_token_details, "reasoning_tokens", 0) or 0),
        cached_prompt_tokens=int(getattr(input_token_details, "cached_tokens", 0) or 0),
    )

    log.debug(
        "openai_usage",
        model=model.value,
        usage=openai_usage.model_dump(),
        cents=openai_usage.cents(model=model),
        finish_reason=getattr(raw_response, "finish_reason", None),
        reasoning_content=getattr(raw_response, "reasoning", None),
    )

    if model in [Model.o3_pro]:
        debug(raw_response.model_dump())

    payload = extract_structured_output(raw_response)
    output: BMType = structure.model_validate(payload)
    return output


def update_messages_xai(messages: list[dict]) -> list:
    final_messages = []
    for message in messages:
        if message["role"] == "system":
            role = system
        elif message["role"] == "user":
            role = user
        elif message["role"] == "assistant":
            role = assistant
        else:
            raise Exception(f"invalid role in message: {message}")
        if isinstance(message["content"], list):
            for c in message["content"]:
                if c["type"] in ["input_text", "output_text"]:
                    final_messages.append(role(c["text"]))
                elif c["type"] == "input_image":
                    final_messages.append(role(image(c["image_url"])))
                else:
                    raise Exception(f"invalid content type: {c}")
        else:
            raise Exception(f"make sure content is a list!: {message}")
    return final_messages


def update_messages_anthropic(messages: list[dict]) -> list[dict]:
    messages = copy.deepcopy(messages)
    for message in messages:
        if "content" in message:
            if isinstance(message["content"], list):
                for c in message["content"]:
                    if c["type"] in ["input_text", "output_text"]:
                        c["type"] = "text"
                    if c["type"] == "input_image":
                        c["type"] = "image"
                        c["source"] = {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": c["image_url"].replace(
                                "data:image/png;base64,", ""
                            ),
                        }
                        del c["image_url"]
                        del c["detail"]
    return messages


MAX_TOKENS_ANTHROPIC_D: dict[Model, int] = {
    Model.sonnet_4: 64_000,
    Model.opus_4: 32_000,
    Model.sonnet_4_5: 64_000,
}
MAX_TOKENS_THINKING_ANTHROPIC_D: dict[Model, int] = {
    Model.sonnet_4: 60_000,
    Model.opus_4: 30_000,
    Model.sonnet_4_5: 60_000,
}
MAX_TOKENS_DEEPSEEK_D: dict[Model, int] = {
    Model.deepseek_chat: 8_192,
    Model.deepseek_reasoner: 32_768,
}


async def _get_next_structure_anthropic(
    structure: type[BMType],  # type[T]
    model: Model,
    messages: list,
) -> BMType:
    tool_schema = structure.model_json_schema()
    messages = update_messages_anthropic(messages=messages)
    response = await anthropic_client.messages.create(
        model=model.value,
        messages=messages,
        max_tokens=MAX_TOKENS_ANTHROPIC_D[model],
        tools=[
            {
                "name": "output_grid",
                "description": tool_schema["description"],
                "input_schema": tool_schema,
            }
        ],
        thinking={
            "type": "enabled",
            "budget_tokens": MAX_TOKENS_THINKING_ANTHROPIC_D[model],
        },
    )
    tool_call = next(block for block in response.content if block.type == "tool_use")
    tool_input = tool_call.input
    output: BMType = structure.model_validate(tool_input)
    return output


class ModelPricing(BaseModel):
    prompt_tokens: float
    reasoning_tokens: float
    completion_tokens: float


MODEL_PRICING_D: dict[Model, ModelPricing] = {
    Model.grok_4: ModelPricing(
        prompt_tokens=300 / 1_000_000,
        reasoning_tokens=1_500 / 1_000_000,
        completion_tokens=1_500 / 1_000_000,
    ),
    Model.grok_3_mini_fast: ModelPricing(
        prompt_tokens=60 / 1_000_000,
        reasoning_tokens=400 / 1_000_000,
        completion_tokens=400 / 1_000_000,
    ),
    # OpenAI pricing (per million tokens)
    Model.o3: ModelPricing(
        prompt_tokens=5_000 / 1_000_000,  # $5 per 1M tokens
        reasoning_tokens=25_000 / 1_000_000,  # $25 per 1M tokens
        completion_tokens=15_000 / 1_000_000,  # $15 per 1M tokens
    ),
    Model.o3_pro: ModelPricing(
        prompt_tokens=1_5_00 / 1_000_000,  # $15 per 1M tokens
        reasoning_tokens=6_000 / 1_000_000,  # $60 per 1M tokens
        completion_tokens=6_000 / 1_000_000,  # $60 per 1M tokens
    ),
    Model.o4_mini: ModelPricing(
        prompt_tokens=300 / 1_000_000,  # $0.30 per 1M tokens
        reasoning_tokens=1_200 / 1_000_000,  # $1.20 per 1M tokens
        completion_tokens=1_200 / 1_000_000,  # $1.20 per 1M tokens
    ),
    Model.gpt_4_1: ModelPricing(
        prompt_tokens=250 / 1_000_000,  # $2.50 per 1M tokens
        reasoning_tokens=1_000 / 1_000_000,  # $10 per 1M tokens
        completion_tokens=1_000 / 1_000_000,  # $10 per 1M tokens
    ),
    Model.gpt_4_1_mini: ModelPricing(
        prompt_tokens=150 / 1_000_000,  # $0.15 per 1M tokens
        reasoning_tokens=600 / 1_000_000,  # $0.60 per 1M tokens
        completion_tokens=600 / 1_000_000,  # $0.60 per 1M tokens
    ),
    Model.gpt_5: ModelPricing(
        prompt_tokens=125 / 1_000_000,  # $10 per 1M tokens (estimate)
        reasoning_tokens=1_000 / 1_000_000,  # $50 per 1M tokens (estimate)
        completion_tokens=1_000 / 1_000_000,  # $30 per 1M tokens (estimate)
    ),
    Model.gpt_52: ModelPricing(
        prompt_tokens=175 / 1_000_000,  # $1.75 per 1M tokens
        reasoning_tokens=1_400 / 1_000_000,  # $14 per 1M tokens (same as output)
        completion_tokens=1_400 / 1_000_000,  # $14 per 1M tokens
    ),
    Model.gpt_5_pro: ModelPricing(
        prompt_tokens=200 / 1_000_000,  # $20 per 1M tokens (estimate)
        reasoning_tokens=1_200 / 1_000_000,  # $60 per 1M tokens (estimate)
        completion_tokens=1_200 / 1_000_000,  # $60 per 1M tokens (estimate)
    ),
    Model.sonnet_4_5: ModelPricing(
        prompt_tokens=3_000 / 1_000_000,  # $10 per 1M tokens (estimate)
        reasoning_tokens=15_000 / 1_000_000,  # $50 per 1M tokens (estimate)
        completion_tokens=15_000 / 1_000_000,  # $30 per 1M tokens (estimate)
    ),
    # Gemini pricing (per million tokens)
    Model.gemini_2_5: ModelPricing(
        prompt_tokens=1_250 / 1_000_000,  # $1.25 per 1M tokens
        reasoning_tokens=10_000 / 1_000_000,  # $10 per 1M tokens (thinking)
        completion_tokens=10_000 / 1_000_000,  # $10 per 1M tokens
    ),
    Model.gemini_2_5_flash_lite: ModelPricing(
        prompt_tokens=75 / 1_000_000,  # $0.075 per 1M tokens
        reasoning_tokens=300 / 1_000_000,  # $0.30 per 1M tokens
        completion_tokens=300 / 1_000_000,  # $0.30 per 1M tokens
    ),
    Model.gemini_3_pro: ModelPricing(
        prompt_tokens=2_500 / 1_000_000,  # $2.50 per 1M tokens (estimate)
        reasoning_tokens=15_000 / 1_000_000,  # $15 per 1M tokens (thinking, estimate)
        completion_tokens=15_000 / 1_000_000,  # $15 per 1M tokens (estimate)
    ),
    # Pydantic AI Gateway - same pricing as direct, gateway is free during beta
    Model.gemini_3_pro_gateway: ModelPricing(
        prompt_tokens=2_500 / 1_000_000,  # $2.50 per 1M tokens (estimate)
        reasoning_tokens=15_000 / 1_000_000,  # $15 per 1M tokens (thinking, estimate)
        completion_tokens=15_000 / 1_000_000,  # $15 per 1M tokens (estimate)
    ),
    # OpenRouter Gemini 3 Pro - $2/M input, $12/M output (from OpenRouter)
    Model.gemini_3_pro_openrouter: ModelPricing(
        prompt_tokens=2_000 / 1_000_000,  # $2 per 1M tokens
        reasoning_tokens=0 / 1_000_000,  # OpenRouter doesn't charge separately for reasoning
        completion_tokens=12_000 / 1_000_000,  # $12 per 1M tokens
    ),
}


class GrokUsage(BaseModel):
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int
    prompt_text_tokens: int
    reasoning_tokens: int
    cached_prompt_text_tokens: int

    def cents(self, model: Model) -> int:
        pricing = MODEL_PRICING_D[model]
        return round(
            self.prompt_tokens * pricing.prompt_tokens
            + self.reasoning_tokens * pricing.reasoning_tokens
            + self.completion_tokens * pricing.completion_tokens
        )


class OpenAIUsage(BaseModel):
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int
    reasoning_tokens: int = 0
    cached_prompt_tokens: int = 0

    def cents(self, model: Model) -> float:
        if model not in MODEL_PRICING_D:
            return 0.0
        pricing = MODEL_PRICING_D[model]
        return round(
            self.prompt_tokens * pricing.prompt_tokens
            + self.reasoning_tokens * pricing.reasoning_tokens
            + self.completion_tokens * pricing.completion_tokens,
            2,
        )


class GeminiUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    thinking_tokens: int = 0
    cached_tokens: int = 0

    def cents(self, model: Model) -> float:
        if model not in MODEL_PRICING_D:
            return 0.0
        pricing = MODEL_PRICING_D[model]
        return round(
            self.prompt_tokens * pricing.prompt_tokens
            + self.thinking_tokens * pricing.reasoning_tokens
            + self.completion_tokens * pricing.completion_tokens,
            2,
        )


@retry_with_backoff(max_retries=20)
async def _get_next_structure_xai(
    structure: type[BMType],  # type[T]
    model: Model,
    messages: list,
) -> BMType:
    messages = update_messages_xai(messages=messages)

    # Configure retry policy for rate limiting
    # use decorator from now on
    custom_retry_policy = json.dumps(
        {
            "methodConfig": [
                {
                    "name": [{}],
                    "retryPolicy": {
                        "maxAttempts": 5,
                        "initialBackoff": "1s",
                        "maxBackoff": "60s",
                        "backoffMultiplier": 2.0,
                        "retryableStatusCodes": [
                            "UNAVAILABLE",
                            "RESOURCE_EXHAUSTED",
                            # "DEADLINE_EXCEEDED",
                        ],
                    },
                }
            ]
        }
    )
    api_keys = os.environ["XAI_API_KEY"].split(",")
    xai_client = XaiAsyncClient(
        api_key=random.choice(api_keys),
        timeout=3_010,
        channel_options=[
            # ("grpc.service_config", custom_retry_policy),
        ],
    )
    chat = xai_client.chat.create(
        model=model.value,
        messages=messages,
        max_tokens=256_000,
        # reasoning_effort="high", # not supported for grok-4
    )
    response, struct = await chat.parse(shape=structure)
    try:
        grok_usage = GrokUsage(
            completion_tokens=response.usage.completion_tokens,
            prompt_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
            prompt_text_tokens=response.usage.prompt_text_tokens,
            reasoning_tokens=response.usage.reasoning_tokens,
            cached_prompt_text_tokens=response.usage.cached_prompt_text_tokens,
        )
        log.debug(
            "usage",
            usage=grok_usage,
            cents=grok_usage.cents(model=model),
            finish_reason=response.finish_reason,
            reasoning_content=response.reasoning_content
            if os.getenv("LOG_GRIDS", "0") == "1"
            else None,
        )

    except Exception as e:
        print(f"usage error: {e=}")
        pass
    return struct


def update_messages_deepseek(
    messages: list[dict], structure: type[BMType]
) -> list[dict]:
    messages = copy.deepcopy(messages)
    schema = structure.model_json_schema()

    # Convert messages to simple format expected by DeepSeek
    final_messages = []

    # Add system message with JSON instructions
    system_content = f"""You are a helpful assistant that outputs structured JSON data.
Always output valid JSON that strictly follows this schema:
{schema}

IMPORTANT: Give the output in a valid JSON string (it should not be wrapped in markdown, just plain json object)."""

    final_messages.append({"role": "system", "content": system_content})

    for message in messages:
        if message["role"] == "system":
            # Append to our system message
            final_messages[0]["content"] += f"\n\n{message.get('content', '')}"
        else:
            if isinstance(message["content"], list):
                # Concatenate all text content
                text_parts = []
                for c in message["content"]:
                    if c["type"] in ["input_text", "output_text"]:
                        text_parts.append(c["text"])
                content = " ".join(text_parts)
            else:
                content = message["content"]

            final_messages.append({"role": message["role"], "content": content})

    return final_messages


async def _get_next_structure_deepseek(
    structure: type[BMType],  # type[T]
    model: Model,
    messages: list,
) -> BMType:
    import json

    messages = update_messages_deepseek(messages=messages, structure=structure)

    # Use JSON mode
    response = await deepseek_client.chat.completions.create(
        model=model.value,
        messages=messages,
        response_format={"type": "json_object"},
        max_tokens=MAX_TOKENS_DEEPSEEK_D[model],
        # temperature=0.3,  # Lower temperature for more consistent JSON output
    )

    # Parse the JSON response
    content = response.choices[0].message.content
    if not content:
        raise Exception("Empty response from DeepSeek model")

    try:
        json_data = json.loads(content)
        output: BMType = structure.model_validate(json_data)
        return output
    except json.JSONDecodeError as e:
        raise Exception(f"Failed to parse JSON response: {e}\nResponse: {content}")


def update_messages_openrouter(
    messages: list[dict], structure: type[BMType] = None, use_json_object: bool = False
) -> list[dict]:
    """Convert messages to OpenRouter format, optionally with schema instructions for json_object mode."""
    messages = copy.deepcopy(messages)
    final_messages = []

    # If using json_object mode (not json_schema), we need to add instructions
    if use_json_object and structure:
        schema = structure.model_json_schema()
        system_content = f"""You are a helpful assistant that outputs structured JSON data.
Always output valid JSON that strictly follows this schema:
{schema}

IMPORTANT: Give the output in a valid JSON string (it should not be wrapped in markdown, just plain json object)."""
        final_messages.append({"role": "system", "content": system_content})

    for message in messages:
        if isinstance(message["content"], list):
            # Handle structured content format
            text_parts = []
            for c in message["content"]:
                if c["type"] in ["input_text", "output_text", "text"]:
                    text_parts.append(c.get("text", c.get("content", "")))
            content = " ".join(text_parts)
        else:
            content = message["content"]

        # If we added a system message for json_object mode, append to it
        if use_json_object and structure and message["role"] == "system":
            final_messages[0]["content"] += f"\n\n{content}"
        else:
            final_messages.append({"role": message["role"], "content": content})

    return final_messages


def update_messages_gemini(messages: list[dict]) -> str:
    """Convert messages to a single prompt string for Gemini."""
    parts = []

    for message in messages:
        role = message["role"]

        if isinstance(message["content"], list):
            # Handle structured content format
            text_parts = []
            for c in message["content"]:
                if c["type"] in ["input_text", "output_text", "text"]:
                    text_parts.append(c.get("text", c.get("content", "")))
                # Note: For now, we're skipping image handling for Gemini
                # You can add image support later if needed
            content = " ".join(text_parts)
        else:
            content = message["content"]

        if role == "system":
            parts.append(f"System: {content}")
        elif role == "user":
            parts.append(f"User: {content}")
        elif role == "assistant":
            parts.append(f"Assistant: {content}")

    return "\n\n".join(parts)


@retry_with_backoff(max_retries=20)
async def _get_next_structure_openrouter(
    structure: type[BMType],  # type[T]
    model: Model,
    messages: list,
) -> BMType:
    import json

    # Check if we need to use json_object mode
    use_json_object = False
    if model in [Model.openrouter_qwen_235b_thinking, Model.openrouter_gpt_oss_120b]:
        use_json_object = True

    messages = update_messages_openrouter(
        messages=messages,
        structure=structure if use_json_object else None,
        use_json_object=use_json_object,
    )

    # Get the JSON schema for the structure
    schema = structure.model_json_schema()

    # Ensure additionalProperties is set to false for strict validation
    if "additionalProperties" not in schema:
        schema["additionalProperties"] = False

    # Default to using json_schema format for structured outputs
    response_format = {
        "type": "json_schema",
        "json_schema": {"name": structure.__name__, "strict": True, "schema": schema},
    }

    extra_body = {}

    # Special cases for certain models
    if model in [Model.openrouter_qwen_235b_thinking]:
        # This model might not support structured outputs
        response_format = {"type": "json_object"}
        extra_body["provider"] = {
            "order": ["Novita"],
            "allow_fallbacks": True,
        }
    elif model in [
        Model.openrouter_qwen_235b,
        # Model.openrouter_qwen_235b_thinking,
    ]:
        extra_body["provider"] = {
            "only": ["cerebras"],
            # "allow_fallbacks": True,
        }
    elif model == Model.openrouter_gpt_oss_120b:
        # Groq doesn't support json_schema, only json_object
        response_format = {"type": "json_object"}
        # Sort providers by throughput for maximum speed
        extra_body["provider"] = {
            # "sort": "throughput",
            "allow_fallbacks": False,
            "only": ["Cerebras", "Groq"],
        }
    elif model == Model.gemini_3_pro_openrouter:
        # Gemini 3 Pro via OpenRouter - force Google provider, enable reasoning
        extra_body["provider"] = {
            "only": ["Google"],
            "allow_fallbacks": False,
        }
        # OpenRouter's reasoning parameter for thinking models
        extra_body["reasoning"] = {"effort": "high"}
    else:
        # Default: sort by throughput for all other OpenRouter models
        extra_body["provider"] = {
            "sort": "throughput",
            "allow_fallbacks": True,
        }
    # if model in [Model.openrouter_glm]:
    #     extra_body["reasoning"]["enabled"] = True

    response = await openrouter_client.chat.completions.create(
        model=model.value,
        messages=messages,
        response_format=response_format,
        max_tokens=100_000,  # Default max tokens for OpenRouter
        # temperature=0.3,  # Lower temperature for more consistent JSON output
        extra_body=extra_body,
        # reasoning_effort="high",
    )

    # Parse the JSON response
    content = response.choices[0].message.content
    if not content:
        # debug(response)
        raise Exception("Empty response from OpenRouter model")

    try:
        json_data = json.loads(content)
        output: BMType = structure.model_validate(json_data)
        return output
    except json.JSONDecodeError as e:
        raise Exception(f"Failed to parse JSON response: {e}\nResponse: {content}")


# Gemini model output token limits
GEMINI_MODEL_MAX_OUTPUT_TOKENS: dict[Model, int] = {
    Model.gemini_2_5: 65_536,
    Model.gemini_2_5_flash_lite: 8_192,
    Model.gemini_3_pro: 65_536,  # Max output tokens
}

# Gemini thinking budget (for reasoning models)
GEMINI_MODEL_THINKING_BUDGET: dict[Model, int] = {
    Model.gemini_3_pro: 65_535,  # Max allowed: 65535
}


@retry_with_backoff(max_retries=20)
async def _get_next_structure_gemini(
    structure: type[BMType],  # type[T]
    model: Model,
    messages: list,
) -> BMType:
    # Convert messages to Gemini format
    prompt = update_messages_gemini(messages=messages)

    # Build config for structured output with maxed out settings
    config: dict[str, T.Any] = {
        "response_mime_type": "application/json",
        "response_schema": structure,
        "max_output_tokens": GEMINI_MODEL_MAX_OUTPUT_TOKENS.get(model, 65_536),
    }

    # Enable thinking for reasoning models (Gemini 3 Pro)
    if model in GEMINI_MODEL_THINKING_BUDGET:
        config["thinking_config"] = {
            "thinking_budget": GEMINI_MODEL_THINKING_BUDGET[model],
        }

    # Use native async API instead of asyncio.to_thread
    response = await gemini_client.aio.models.generate_content(
        model=model.value,
        contents=prompt,
        config=config,
    )

    # Extract and log usage metadata
    usage_metadata = response.usage_metadata
    if usage_metadata:
        gemini_usage = GeminiUsage(
            prompt_tokens=usage_metadata.prompt_token_count or 0,
            completion_tokens=usage_metadata.candidates_token_count or 0,
            total_tokens=usage_metadata.total_token_count or 0,
            thinking_tokens=usage_metadata.thoughts_token_count or 0,
            cached_tokens=usage_metadata.cached_content_token_count or 0,
        )
        log.debug(
            "gemini_usage",
            model=model.value,
            usage=gemini_usage.model_dump(),
            cents=gemini_usage.cents(model=model),
        )

    # The response.parsed should contain the instantiated object
    if hasattr(response, "parsed") and response.parsed:
        return response.parsed

    # Fallback to parsing the text response
    import json

    content = response.text
    if not content:
        raise Exception("Empty response from Gemini model")

    try:
        json_data = json.loads(content)
        output: BMType = structure.model_validate(json_data)
        return output
    except json.JSONDecodeError as e:
        raise Exception(f"Failed to parse JSON response: {e}\nResponse: {content}")


def update_messages_pydantic_ai(messages: list[dict]) -> str:
    """Convert messages to a single prompt string for Pydantic AI Agent.run()."""
    parts = []

    for message in messages:
        role = message["role"]

        if isinstance(message["content"], list):
            # Handle structured content format (OpenAI-style)
            text_parts = []
            for c in message["content"]:
                if c["type"] in ["input_text", "output_text", "text"]:
                    text_parts.append(c.get("text", c.get("content", "")))
            content = " ".join(text_parts)
        else:
            content = message["content"]

        if role == "system":
            parts.append(f"System: {content}")
        elif role == "user":
            parts.append(f"User: {content}")
        elif role == "assistant":
            parts.append(f"Assistant: {content}")

    return "\n\n".join(parts)


# Pydantic AI Gateway settings for Gemini models (Vertex AI limits)
# Vertex AI has different limits than direct Gemini API: thinking_budget 1-32768
PYDANTIC_GATEWAY_THINKING_BUDGET: dict[Model, int] = {
    Model.gemini_3_pro_gateway: 32_768,  # Max for Vertex AI
}

PYDANTIC_GATEWAY_MAX_TOKENS: dict[Model, int] = {
    Model.gemini_3_pro_gateway: 65_536,  # Max output tokens
}


@retry_with_backoff(max_retries=20)
async def _get_next_structure_pydantic_gateway(
    structure: type[BMType],
    model: Model,
    messages: list,
) -> BMType:
    """
    Use Pydantic AI Gateway to call Gemini models with native thinking_config support.
    Requires PYDANTIC_AI_GATEWAY_API_KEY environment variable.
    """
    # Build model settings with thinking config
    thinking_budget = PYDANTIC_GATEWAY_THINKING_BUDGET.get(model, 65_535)
    max_tokens = PYDANTIC_GATEWAY_MAX_TOKENS.get(model, 65_536)

    settings = GoogleModelSettings(
        max_tokens=max_tokens,
        google_thinking_config={"thinking_budget": thinking_budget},
    )

    # Create gateway provider with explicit API key
    # Support both env var names
    gateway_api_key = os.environ.get("PYDANTIC_AI_GATEWAY_API_KEY") or os.environ.get("PYDANTIC_API_GATEWAY_API_KEY")
    if not gateway_api_key:
        raise ValueError("Set PYDANTIC_AI_GATEWAY_API_KEY or PYDANTIC_API_GATEWAY_API_KEY environment variable")
    
    # Create custom HTTP client with long timeout for reasoning models (3 hours like GPT-5-Pro)
    http_client = httpx.AsyncClient(timeout=httpx.Timeout(10_800.0))
    
    # gateway_provider takes upstream provider as string (e.g., "google-vertex")
    gateway = gateway_provider("google-vertex", api_key=gateway_api_key, http_client=http_client)

    # Extract model name from the gateway path (e.g., "gateway/google-vertex:gemini-3-pro-preview" -> "gemini-3-pro-preview")
    model_name = model.value.split(":")[-1]
    google_model = GoogleModel(model_name, provider=gateway)

    # Create agent with structured output type
    agent: Agent[None, BMType] = Agent(
        google_model,
        output_type=structure,
        model_settings=settings,
    )

    # Convert messages to prompt string
    prompt = update_messages_pydantic_ai(messages)

    # Run the agent
    result = await agent.run(prompt)

    # Log usage if available
    usage = result.usage()
    if usage:
        gemini_usage = GeminiUsage(
            prompt_tokens=usage.request_tokens or 0,
            completion_tokens=usage.response_tokens or 0,
            total_tokens=usage.total_tokens or 0,
            thinking_tokens=0,  # Pydantic AI doesn't expose thinking tokens separately yet
            cached_tokens=0,
        )
        log.debug(
            "pydantic_gateway_usage",
            model=model.value,
            usage=gemini_usage.model_dump(),
            cents=gemini_usage.cents(model=model),
        )

    return result.output


async def main_test() -> None:
    class Reasoning(BaseModel):
        """Reasoning over a problem returning the reasoning string and the answer."""

        reasoning: str = Field(..., description="Reasoning for the math problem.")
        answer: int = Field(..., description="The answer to the math problem.")

    # test openai
    """
    response = await _get_next_structure_openai(
        structure=Reasoning,
        model=Model.o4_mini,
        messages=[
            {
                "role": "system",
                "content": "you are a math solving pirate. always talk like a pirate.",
            },
            {"role": "user", "content": "what is 39 * 28937?"},
        ],
    )
    debug(response)

    # test anthropic
    response = await _get_next_structure_anthropic(
        structure=Reasoning,
        model=Model.sonnet_4,
        messages=[
            {
                "role": "user",
                "content": "you are a math solving pirate. always talk like a pirate.",
            },
            {"role": "user", "content": "what is 39 * 28937?"},
        ],
    )
    debug(response)
    """
    # test groq with structured outputs
    response = await _get_next_structure_openrouter(
        structure=Reasoning,
        model=Model.openrouter_gpt_oss_120b,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "you are a math solving pirate. always talk like a pirate.",
                    }
                ],
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "what is 39 * 28937?"}],
            },
        ],
    )
    debug(response)

    # test gemini
    # response = await get_next_structure(
    #     structure=Reasoning,
    #     model=Model.gemini_2_5_flash_lite,
    #     messages=[
    #         {
    #             "role": "system",
    #             "content": "you are a math solving pirate. always talk like a pirate.",
    #         },
    #         {"role": "user", "content": "what is 39 * 28937?"},
    #     ],
    # )
    # debug(response)


if __name__ == "__main__":
    asyncio.run(main_test())
