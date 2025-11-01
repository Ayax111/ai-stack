from __future__ import annotations

import time
from typing import Optional, Tuple
from urllib.parse import urlparse

import requests

DEFAULT_HOST = "192.168.0.194"
DEFAULT_CHAT_PORT = 1234
DEFAULT_EMBED_PORT = 1235


class LMStudioManager:
    """
    Minimal helper around the experimental LM Studio management endpoints.

    It is capable of:
    - verifying connectivity (`/v1/models`)
    - checking which models are loaded
    - requesting model loads/unloads (if supported by your build)
    - issuing lightweight test prompts
    """

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_CHAT_PORT,
        base_url: Optional[str] = None,
        token: Optional[str] = None,
    ):
        self.host, self.port, self.base_url = self._resolve_base(host, port, base_url)
        self.headers = {"Authorization": f"Bearer {token}"} if token else {}

    @staticmethod
    def _resolve_base(host: str, port: int, base_url: Optional[str]) -> Tuple[str, int, str]:
        if base_url:
            parsed = urlparse(base_url if "://" in base_url else f"http://{base_url}")
            resolved_host = parsed.hostname or host or DEFAULT_HOST
            resolved_port = parsed.port or port or DEFAULT_CHAT_PORT
            scheme = parsed.scheme or "http"
            netloc = resolved_host if parsed.port is None else f"{resolved_host}:{resolved_port}"
            path = parsed.path.rstrip("/") or "/v1"
            return resolved_host, resolved_port, f"{scheme}://{netloc}{path}"
        resolved_host = host or DEFAULT_HOST
        resolved_port = port or DEFAULT_CHAT_PORT
        return resolved_host, resolved_port, f"http://{resolved_host}:{resolved_port}/v1"

    def check_connection(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/models", timeout=5, headers=self.headers)
            return response.status_code == 200
        except requests.exceptions.RequestException as exc:
            print(f"❌ Cannot connect to LM Studio: {exc}")
            return False

    def get_loaded_models(self):
        try:
            response = requests.get(f"{self.base_url}/models", timeout=10, headers=self.headers)
            if response.status_code == 200:
                payload = response.json()
                return [model["id"] for model in payload.get("data", [])]
            return []
        except Exception as exc:  # pragma: no cover - best effort logging
            print(f"Error getting models: {exc}")
            return []

    def ensure_model_loaded(self, model_name: str, wait_time: int = 60) -> bool:
        if not model_name:
            raise ValueError("Model name must be provided")
        if model_name in self.get_loaded_models():
            return True
        try:
            print(f"[LMStudioManager] Solicitud de carga → {self.base_url} (modelo={model_name})")
            response = requests.post(
                f"{self.base_url}/models/load",
                json={"model": model_name},
                timeout=10,
                headers=self.headers,
            )
        except requests.exceptions.RequestException as exc:
            print(f"❌ Error loading model {model_name}: {exc}")
            return False
        if response.status_code != 200:
            print(f"❌ Error loading model {model_name}: {response.status_code}")
            print(f"Response: {response.text}")
            return False
        if wait_time > 0:
            print(f"⏳ Waiting {wait_time}s for {model_name} to load...")
            time.sleep(wait_time)
        return model_name in self.get_loaded_models()

    def load_model(self, model_name: str, wait_time: int = 60) -> bool:
        print(f"🔄 Loading model: {model_name}")
        return self.ensure_model_loaded(model_name, wait_time)

    def unload_model(self, model_name: str) -> bool:
        try:
            response = requests.post(
                f"{self.base_url}/models/unload",
                json={"model": model_name},
                headers=self.headers,
            )
        except requests.exceptions.RequestException as exc:
            print(f"Error unloading model: {exc}")
            return False
        if response.status_code == 200:
            print(f"✅ Model {model_name} unloaded")
            return True
        return False

    def test_model(self, model_name: str, prompt: str = "Hello, does it work correctly?") -> bool:
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                json={
                    "model": model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.7,
                    "max_tokens": 100,
                },
                timeout=60,
                headers=self.headers,
            )
        except requests.exceptions.RequestException as exc:
            print(f"Error testing model: {exc}")
            return False
        if response.status_code != 200:
            print(f"❌ Error in test: {response.status_code}")
            return False
        result = response.json()
        answer = result["choices"][0]["message"]["content"]
        print(f"🤖 Model response: {answer}")
        return True


def normalize_lmstudio_url(
    primary: Optional[str],
    *,
    fallback: Optional[str] = None,
    default_host: str = DEFAULT_HOST,
    default_port: int = DEFAULT_CHAT_PORT,
    override_port: Optional[int] = None,
    override_host: Optional[str] = None,
    default_path: str = "/v1",
) -> str:
    """
    Normaliza una URL base de LM Studio garantizando esquema, puerto y path.

    - Si `primary` está definido se usa como fuente principal.
    - En caso contrario se toma `fallback`.
    - Si ninguna está definida se genera a partir de `default_host` + `default_port`.
    - `override_port` permite forzar un puerto concreto (p. ej., 1235 para embeddings).
    """

    raw = primary or fallback
    if raw:
        parsed = urlparse(raw if "://" in raw else f"http://{raw}")
    else:
        parsed = urlparse(f"http://{default_host}:{default_port}{default_path}")

    scheme = parsed.scheme or "http"
    host = override_host or parsed.hostname or default_host
    path = parsed.path.rstrip("/") or default_path

    raw_port = parsed.port
    if override_port is not None:
        port = override_port
    else:
        port = raw_port if raw_port is not None else default_port

    print(
        f"[LMStudioManager] normalize_lmstudio_url → base={primary or fallback} | "
        f"override_host={override_host or host} | resolved={scheme}://{host}:{port}{path}"
    )

    netloc = host if port is None else f"{host}:{port}"
    return f"{scheme}://{netloc}{path}"
