# ~/ai-stack/rag/llm_client.py
import os

from openai import OpenAI

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://192.168.0.194:1234/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "lm-studio")
LLM_MODEL = os.getenv("LLM_MODEL", "mistralai/magistral-small-2509")

_client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)


def chat(system_prompt: str, user_prompt: str, temperature: float = 0.2) -> str:
    if not LLM_MODEL:
        raise RuntimeError("Configura LLM_MODEL en .env (mira /v1/models de LM Studio).")
    resp = _client.chat.completions.create(
        model=LLM_MODEL,
        messages=([{"role": "system", "content": system_prompt}] if system_prompt else [])
        + [{"role": "user", "content": user_prompt}],
        temperature=temperature,
    )
    return resp.choices[0].message.content
