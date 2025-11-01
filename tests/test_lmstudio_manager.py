from unittest.mock import patch

from LMStudioManager import (
    DEFAULT_EMBED_PORT,
    LMStudioManager,
    normalize_lmstudio_url,
)


class DummyResponse:
    def __init__(self, status_code=200, json_payload=None, text=""):
        self.status_code = status_code
        self._json = json_payload or {}
        self.text = text or ""

    def json(self):
        return self._json


def make_list_response(model_ids):
    return DummyResponse(
        200,
        {
            "data": [
                {"id": mid, "object": "model", "owned_by": "organization_owner"}
                for mid in model_ids
            ]
        },
    )


def test_normalize_switches_to_embed_port_when_matching_chat():
    """If both URLs point to the chat port, the helper should prefer the embed port."""
    fallback = "http://192.168.0.194:1234/v1"
    primary = "http://192.168.0.194:1234/v1"

    result = normalize_lmstudio_url(
        primary,
        fallback=fallback,
        default_host="192.168.0.194",
        default_port=DEFAULT_EMBED_PORT,
        override_port=DEFAULT_EMBED_PORT,
    )

    assert result == "http://192.168.0.194:1235/v1"


def test_normalize_respects_explicit_embed_host_and_port():
    result = normalize_lmstudio_url(
        None,
        fallback="http://192.168.0.194:1234/v1",
        default_host="192.168.0.194",
        default_port=DEFAULT_EMBED_PORT,
        override_port=9999,
        override_host="10.0.0.5",
    )

    assert result == "http://10.0.0.5:9999/v1"


@patch("LMStudioManager.time.sleep", autospec=True)
@patch("LMStudioManager.requests.post", autospec=True)
@patch("LMStudioManager.requests.get", autospec=True)
def test_ensure_model_loaded_returns_true_when_already_loaded(mock_get, mock_post, mock_sleep):
    mock_get.return_value = make_list_response(["chat-model", "embedding-model"])

    mgr = LMStudioManager(base_url="http://localhost:1234/v1")
    assert mgr.ensure_model_loaded("embedding-model", wait_time=1) is True
    mock_post.assert_not_called()
    mock_sleep.assert_not_called()


@patch("LMStudioManager.time.sleep", autospec=True)
@patch("LMStudioManager.requests.post", autospec=True)
@patch("LMStudioManager.requests.get", autospec=True)
def test_ensure_model_loaded_triggers_load_when_missing(mock_get, mock_post, mock_sleep):
    # First call -> model missing, second call -> model present
    mock_get.side_effect = [
        make_list_response([]),
        make_list_response(["embedding-model"]),
    ]
    mock_post.return_value = DummyResponse(200)

    mgr = LMStudioManager(base_url="http://localhost:1234/v1")
    assert mgr.ensure_model_loaded("embedding-model", wait_time=1) is True

    assert mock_post.call_count == 1
    payload = mock_post.call_args.kwargs.get("json")
    assert payload == {"model": "embedding-model"}
    # Wait hook invoked with supplied wait time
    mock_sleep.assert_called_once_with(1)


@patch("LMStudioManager.requests.post", autospec=True)
@patch("LMStudioManager.requests.get", autospec=True)
def test_ensure_model_loaded_returns_false_on_error(mock_get, mock_post):
    mock_get.return_value = make_list_response([])
    mock_post.return_value = DummyResponse(500, text="boom")

    mgr = LMStudioManager(base_url="http://localhost:1234/v1")
    assert mgr.ensure_model_loaded("embedding-model", wait_time=0) is False
