"""环境变量与 LLM 配置（OpenAI 兼容协议接入 DeepSeek）。"""

from __future__ import annotations

import os
from pathlib import Path

# 关闭 CrewAI 1.x 的 OpenTelemetry 遥测（必须在导入 crewai 之前设置）
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

from dotenv import load_dotenv  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
BASE_URL = os.getenv("OPENAI_API_BASE", "https://api.deepseek.com/v1").strip()
MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-chat").strip()

# .env 中的占位符前缀，用于判断用户是否已填入真实密钥
_PLACEHOLDER_PREFIX = ("sk-在这里", "sk-xxxx", "your-", "填写")


def is_api_key_configured() -> bool:
    """判断用户是否已在 .env 中填入真实 API Key。"""
    if not API_KEY:
        return False
    return not any(API_KEY.startswith(p) for p in _PLACEHOLDER_PREFIX)


def get_llm(temperature: float = 0.3, max_tokens: int = 8192):
    """构建 CrewAI LLM 实例（经 litellm 以 OpenAI 兼容协议访问 DeepSeek）。"""
    if not is_api_key_configured():
        raise RuntimeError(
            "未检测到有效的 DeepSeek API Key。\n"
            f"请编辑 {PROJECT_ROOT / '.env'}，将 OPENAI_API_KEY 替换为你的真实密钥 "
            "（申请地址：https://platform.deepseek.com/）。"
        )

    from crewai import LLM

    # custom_openai=True：使用 CrewAI 1.x 内置的原生 OpenAI SDK，
    # 配合自定义 base_url 接入 DeepSeek（或任意 OpenAI 兼容服务），无需 litellm
    return LLM(
        model=MODEL_NAME,
        base_url=BASE_URL,
        api_key=API_KEY,
        custom_openai=True,
        temperature=temperature,
        max_tokens=max_tokens,
    )
