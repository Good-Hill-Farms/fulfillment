"""
Model constants for the fulfillment-mixy-matchi application.
Contains model identifiers for OpenAI, Anthropic, Google, and Perplexity models.
"""

# OpenAI Models
GPT_4_1 = "openai/gpt-4.1"
GPT_4_1_MINI = "openai/gpt-4.1-mini"
GPT_4_1_NANO = "openai/gpt-4.1-nano"
O1_PRO = "openai/o1-pro"
GPT_4O_MINI_SEARCH = "openai/gpt-4o-mini-search-preview"
GPT_4O_SEARCH = "openai/gpt-4o-search-preview"
GPT_4_5_PREVIEW = "openai/gpt-4.5-preview"
O1 = "openai/o1"
GPT_4O_2024_11_20 = "openai/gpt-4o-2024-11-20"
O1_MINI = "openai/o1-mini"
O1_PREVIEW = "openai/o1-preview"
CHATGPT_4O_LATEST = "openai/chatgpt-4o-latest"
GPT_4O_MINI = "openai/gpt-4o-mini"
GPT_4O = "openai/gpt-4o"
GPT_4O_EXTENDED = "openai/gpt-4o:extended"
GPT_4_TURBO = "openai/gpt-4-turbo"

# Anthropic Models
CLAUDE_3_7_SONNET = "anthropic/claude-3.7-sonnet"
CLAUDE_3_7_SONNET_BETA = "anthropic/claude-3.7-sonnet:beta"
CLAUDE_3_5_HAIKU = "anthropic/claude-3.5-haiku"
CLAUDE_3_5_HAIKU_BETA = "anthropic/claude-3.5-haiku:beta"
CLAUDE_3_5_SONNET = "anthropic/claude-3.5-sonnet"
CLAUDE_3_5_SONNET_BETA = "anthropic/claude-3.5-sonnet:beta"
CLAUDE_3_OPUS = "anthropic/claude-3-opus"
CLAUDE_3_OPUS_BETA = "anthropic/claude-3-opus:beta"
CLAUDE_3_SONNET = "anthropic/claude-3-sonnet"
CLAUDE_3_HAIKU = "anthropic/claude-3-haiku"

# Google Models
GEMINI_2_5_PRO_PREVIEW = "google/gemini-2.5-pro-preview-03-25"
GEMINI_2_5_FLASH_PREVIEW = "google/gemini-2.5-flash-preview"
GEMINI_2_5_PRO_EXP = "google/gemini-2.5-pro-exp-03-25:free"
GEMINI_2_0_FLASH_LITE = "google/gemini-2.0-flash-lite-001"
GEMINI_2_0_FLASH = "google/gemini-2.0-flash-001"
GEMINI_2_0_FLASH_EXP = "google/gemini-2.0-flash-exp:free"
GEMINI_PRO_1_5 = "google/gemini-pro-1.5"
GEMINI_PRO = "google/gemini-pro"
GEMINI_PRO_VISION = "google/gemini-pro-vision"

# Perplexity Models
PERPLEXITY_SONAR = "perplexity/sonar-reasoning"
PERPLEXITY_SONAR_REASONING_PRO = "perplexity/sonar-reasoning_pro"
PERPLEXITY_SONAR_DEEP_RESEARCH = "perplexity/sonar-deep-research"
PERPLEXITY_SONAR_PRO = "perplexity/sonar-pro"

# Model display names mapping
MODEL_DISPLAY_NAMES = {
    # OpenAI
    GPT_4_1: "GPT-4.1",
    GPT_4_1_MINI: "GPT-4.1 Mini",
    GPT_4_1_NANO: "GPT-4.1 Nano",
    O1_PRO: "O1 Pro",
    GPT_4O_MINI_SEARCH: "GPT-4o Mini Search",
    GPT_4O_SEARCH: "GPT-4o Search",
    GPT_4_5_PREVIEW: "GPT-4.5 Preview",
    O1: "O1",
    GPT_4O_2024_11_20: "GPT-4o (2024-11-20)",
    O1_MINI: "O1 Mini",
    O1_PREVIEW: "O1 Preview",
    CHATGPT_4O_LATEST: "ChatGPT-4o Latest",
    GPT_4O_MINI: "GPT-4o Mini",
    GPT_4O: "GPT-4o",
    GPT_4O_EXTENDED: "GPT-4o Extended",
    GPT_4_TURBO: "GPT-4 Turbo",
    # Anthropic
    CLAUDE_3_7_SONNET: "Claude 3.7 Sonnet",
    CLAUDE_3_7_SONNET_BETA: "Claude 3.7 Sonnet Beta",
    CLAUDE_3_5_HAIKU: "Claude 3.5 Haiku",
    CLAUDE_3_5_HAIKU_BETA: "Claude 3.5 Haiku Beta",
    CLAUDE_3_5_SONNET: "Claude 3.5 Sonnet",
    CLAUDE_3_5_SONNET_BETA: "Claude 3.5 Sonnet Beta",
    CLAUDE_3_OPUS: "Claude 3 Opus",
    CLAUDE_3_OPUS_BETA: "Claude 3 Opus Beta",
    CLAUDE_3_SONNET: "Claude 3 Sonnet",
    CLAUDE_3_HAIKU: "Claude 3 Haiku",
    # Google
    GEMINI_2_5_PRO_PREVIEW: "Gemini 2.5 Pro Preview",
    GEMINI_2_5_FLASH_PREVIEW: "Gemini 2.5 Flash Preview",
    GEMINI_2_5_PRO_EXP: "Gemini 2.5 Pro Exp",
    GEMINI_2_0_FLASH_LITE: "Gemini 2.0 Flash Lite",
    GEMINI_2_0_FLASH: "Gemini 2.0 Flash",
    GEMINI_2_0_FLASH_EXP: "Gemini 2.0 Flash Exp",
    GEMINI_PRO_1_5: "Gemini Pro 1.5",
    GEMINI_PRO: "Gemini Pro",
    GEMINI_PRO_VISION: "Gemini Pro Vision",
    # Perplexity
    PERPLEXITY_SONAR: "Perplexity Sonar Reasoning",
    PERPLEXITY_SONAR_REASONING_PRO: "Perplexity Sonar Reasoning Pro",
    PERPLEXITY_SONAR_DEEP_RESEARCH: "Perplexity Sonar Deep Research",
    PERPLEXITY_SONAR_PRO: "Perplexity Sonar Pro",
}

# Group models by provider for UI organization
MODEL_GROUPS = {
    "OpenAI": [
        GPT_4_1,
        GPT_4_1_MINI,
        GPT_4_1_NANO,
        O1_PRO,
        GPT_4O_MINI_SEARCH,
        GPT_4O_SEARCH,
        GPT_4_5_PREVIEW,
        O1,
        GPT_4O_2024_11_20,
        O1_MINI,
        O1_PREVIEW,
        CHATGPT_4O_LATEST,
        GPT_4O_MINI,
        GPT_4O,
        GPT_4O_EXTENDED,
        GPT_4_TURBO,
    ],
    "Anthropic": [
        CLAUDE_3_7_SONNET,
        CLAUDE_3_7_SONNET_BETA,
        CLAUDE_3_5_HAIKU,
        CLAUDE_3_5_HAIKU_BETA,
        CLAUDE_3_5_SONNET,
        CLAUDE_3_5_SONNET_BETA,
        CLAUDE_3_OPUS,
        CLAUDE_3_OPUS_BETA,
        CLAUDE_3_SONNET,
        CLAUDE_3_HAIKU,
    ],
    "Google": [
        GEMINI_2_5_PRO_PREVIEW,
        GEMINI_2_5_FLASH_PREVIEW,
        GEMINI_2_5_PRO_EXP,
        GEMINI_2_0_FLASH_LITE,
        GEMINI_2_0_FLASH,
        GEMINI_2_0_FLASH_EXP,
        GEMINI_PRO_1_5,
        GEMINI_PRO,
        GEMINI_PRO_VISION,
    ],
    "Perplexity": [
        PERPLEXITY_SONAR,
        PERPLEXITY_SONAR_REASONING_PRO,
        PERPLEXITY_SONAR_DEEP_RESEARCH,
        PERPLEXITY_SONAR_PRO,
    ],
}
