# ~/ai-stack/rag/llm_client.py
import os

from LMStudioManager import LMStudioManager
from openai import OpenAI

_client = None
_manager = None
_client_conf = None
_model_ready: dict[str, bool] = {}

DEFAULT_BASE_URL = "http://192.168.0.194:1234/v1"
DEFAULT_API_KEY = "lm-studio"  # pragma: allowlist secret


def _get_config() -> tuple[OpenAI, LMStudioManager, str]:
    global _client, _manager, _client_conf
    base = os.getenv("LLM_BASE_URL", DEFAULT_BASE_URL)
    token = os.getenv("LLM_API_KEY", DEFAULT_API_KEY)
    model = os.getenv("LLM_MODEL")
    if not model:
        raise RuntimeError("Configura LLM_MODEL en .env (mira /v1/models de LM Studio).")
    conf = (base, token)
    if _client is None or _client_conf != conf:
        _client = OpenAI(base_url=base, api_key=token)  # pragma: allowlist secret
        _manager = LMStudioManager(base_url=base, token=token)  # pragma: allowlist secret
        _client_conf = conf
    return _client, _manager, model


def _ensure_model_ready(manager: LMStudioManager, model: str) -> None:
    if _model_ready.get(model):
        return
    ok = manager.ensure_model_loaded(model, wait_time=90)
    if not ok:
        raise RuntimeError(
            f"No se pudo cargar el modelo {model} en LM Studio. "
            "Verifica que el servidor esté activo y que la API soporte /models/load."
        )
    _model_ready[model] = True


def chat(system_prompt: str, user_prompt: str, temperature: float = 0.2) -> str:
    client, manager, model = _get_config()
    _ensure_model_ready(manager, model)
    resp = client.chat.completions.create(
        model=model,
        messages=([{"role": "system", "content": system_prompt}] if system_prompt else [])
        + [{"role": "user", "content": user_prompt}],
        temperature=temperature,
    )
    return resp.choices[0].message.content
