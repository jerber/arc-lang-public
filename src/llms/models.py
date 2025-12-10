from enum import Enum

from pydantic import BaseModel


class Model(str, Enum):
    gpt_4_5 = "gpt-4.5-preview-2025-02-27"
    gpt_4_o = "gpt-4o-2024-11-20"
    o3_mini = "o3-mini-2025-01-31"
    o3_mini_high = "o3-mini-2025-01-31_high"
    o4_mini_high = "o4-mini_high"
    o4_mini = "o4-mini"
    o3 = "o3"
    o3_pro = "o3-pro"
    gpt_4_1 = "gpt-4.1"
    gpt_4_1_mini = "gpt-4.1-mini"
    gpt_5 = "gpt-5"
    gpt_5_pro = "gpt-5-pro"

    sonnet_3_7 = "claude-3-7-sonnet-latest"
    sonnet_3_5 = "claude-3-5-sonnet-latest"
    sonnet_4_5 = "claude-sonnet-4-5-20250929"
    gemini_2_5 = "gemini-2.5-pro"
    gemini_2_5_flash_lite = "gemini-2.0-flash-lite"
    gemini_3_pro = "gemini-3-pro-preview"

    deepseek_chat = "deepseek-chat"
    deepseek_reasoner = "deepseek-reasoner"

    sonnet_4 = "claude-sonnet-4-20250514"
    opus_4 = "claude-opus-4-20250514"

    grok_4 = "grok-4"
    grok_3_mini_fast = "grok-3-mini-fast"

    openrouter_sonnet_3_7_thinking = "anthropic/claude-3.7-sonnet:thinking"
    openrouter_sonnet_3_7 = "anthropic/claude-3.7-sonnet"
    openrouter_gemini_2_5_free = "google/gemini-2.5-pro-exp-03-25:free"
    openrouter_gemini_2_5 = "google/gemini-2.5-pro-preview-03-25"
    openrouter_deepseek_3_free = "deepseek/deepseek-chat-v3-0324:free"
    openrouter_deepseek_r1 = "deepseek/deepseek-r1"
    openrouter_deepseek_r1_free = "deepseek/deepseek-r1:free"
    openrouter_grok_v3 = "x-ai/grok-3-beta"
    openrouter_quasar_alpha = "openrouter/quasar-alpha"
    openrouter_optimus_alpha = "openrouter/optimus-alpha"

    openrouter_qwen_235b = "qwen/qwen3-235b-a22b"
    openrouter_qwen_235b_thinking = "qwen/qwen3-235b-a22b-thinking-2507"
    openrouter_gemini_2_5_flash_lite = "google/gemini-2.5-flash-lite"
    openrouter_grok_4 = "x-ai/grok-4"

    openrouter_glm = "z-ai/glm-4.5"
    openrouter_kimi_k2 = "moonshotai/kimi-k2"

    openrouter_horizon_alpha = "openrouter/horizon-alpha"

    openrouter_gpt_oss_120b = "openai/gpt-oss-120b"

    cerebras_gpt_oss_120b = "gpt-oss-120b"
    groq_gpt_oss_120b = "openai/gpt-oss-120b"


class ModelConfig(BaseModel):
    max_tokens: int
    max_thinking_tokens: int | None


model_config: dict[Model, ModelConfig] = {
    Model.sonnet_3_5: ModelConfig(max_tokens=8_192, max_thinking_tokens=None),
    Model.sonnet_3_7: ModelConfig(max_tokens=50_000, max_thinking_tokens=30_000),
    Model.sonnet_4_5: ModelConfig(max_tokens=60_000, max_thinking_tokens=60_000),
    Model.gpt_5: ModelConfig(max_tokens=1_000_000, max_thinking_tokens=None),
    Model.gpt_5_pro: ModelConfig(max_tokens=4_000_000, max_thinking_tokens=None),
    Model.gemini_3_pro: ModelConfig(max_tokens=1_000_000, max_thinking_tokens=65_535),
}
