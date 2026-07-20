"""API tests.

These tests run WITHOUT downloading any model: the app is started with
SKIP_MODEL_LOAD=1 and a stub generator is injected where a loaded model is
needed. This keeps the suite fast and CI-friendly while still exercising the
full request/response path, validation, and error handling.
"""

import os

os.environ["SKIP_MODEL_LOAD"] = "1"

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.model import GenerationError


class StubGenerator:
    """Stands in for a loaded TextGenerator."""

    model_id = "stub-model"
    is_loaded = True

    def __init__(self, reply: str = "stub reply", fail: bool = False) -> None:
        self.reply = reply
        self.fail = fail
        self.calls: list[dict] = []

    def generate(self, prompt: str, max_new_tokens: int, temperature: float) -> str:
        self.calls.append(
            {
                "prompt": prompt,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
            }
        )
        if self.fail:
            raise GenerationError("boom")
        return self.reply


@pytest.fixture()
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_reports_model_not_loaded(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is False


def test_generate_returns_503_before_model_loads(client):
    resp = client.post("/generate", json={"prompt": "hello"})
    assert resp.status_code == 503
    assert resp.json()["error"] == "model_not_loaded"


def test_generate_happy_path_returns_clean_json(client):
    stub = StubGenerator(reply="Hello from the model!")
    app.state.generator = stub

    resp = client.post(
        "/generate",
        json={"prompt": "Say hello", "max_new_tokens": 32, "temperature": 0.5},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["generated_text"] == "Hello from the model!"
    assert body["model"] == "stub-model"
    assert body["prompt_chars"] == len("Say hello")
    assert body["generated_chars"] == len("Hello from the model!")
    assert isinstance(body["elapsed_ms"], int)
    # Parameters were passed through to the generator as sent.
    assert stub.calls == [
        {"prompt": "Say hello", "max_new_tokens": 32, "temperature": 0.5}
    ]


def test_generate_uses_documented_defaults(client):
    stub = StubGenerator()
    app.state.generator = stub

    resp = client.post("/generate", json={"prompt": "defaults please"})
    assert resp.status_code == 200
    assert stub.calls[0]["max_new_tokens"] == 128
    assert stub.calls[0]["temperature"] == 0.7


@pytest.mark.parametrize(
    "payload",
    [
        {},  # missing prompt
        {"prompt": ""},  # empty prompt
        {"prompt": "   \n\t "},  # whitespace-only prompt
        {"prompt": "x" * 8001},  # prompt too long
        {"prompt": "ok", "max_new_tokens": 0},  # below minimum
        {"prompt": "ok", "max_new_tokens": 10_000},  # above maximum
        {"prompt": "ok", "temperature": -0.1},  # invalid temperature
        {"prompt": "ok", "temperature": 5},  # invalid temperature
        {"prompt": "ok", "max_tokens": 5},  # unknown field (typo) rejected
    ],
)
def test_generate_rejects_invalid_input_with_422(client, payload):
    app.state.generator = StubGenerator()
    resp = client.post("/generate", json=payload)
    assert resp.status_code == 422


def test_generation_failure_returns_clean_500_json(client):
    app.state.generator = StubGenerator(fail=True)
    resp = client.post("/generate", json={"prompt": "trigger failure"})
    assert resp.status_code == 500
    body = resp.json()
    assert body["error"] == "generation_failed"
    assert "boom" in body["detail"]


def test_ready_returns_503_until_model_loads(client):
    resp = client.get("/ready")
    assert resp.status_code == 503
    assert resp.json()["error"] == "model_not_loaded"


def test_ready_returns_200_once_loaded(client):
    app.state.generator = StubGenerator()
    resp = client.get("/ready")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ready"


def test_concurrent_requests_all_succeed(client):
    """Semaphore queues requests rather than dropping them."""
    from concurrent.futures import ThreadPoolExecutor

    app.state.generator = StubGenerator(reply="ok")
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(
            pool.map(
                lambda i: client.post("/generate", json={"prompt": f"req {i}"}),
                range(8),
            )
        )
    assert all(r.status_code == 200 for r in results)
    assert len(app.state.generator.calls) == 8


def test_saturated_queue_returns_503_server_busy(client, monkeypatch):
    """With capacity 1 held by a slow request, a short queue timeout yields 503."""
    import time as _time
    from concurrent.futures import ThreadPoolExecutor

    monkeypatch.setenv("GENERATION_QUEUE_TIMEOUT_S", "0.2")
    slow = StubGenerator(reply="slow done")
    slow.generate = lambda prompt, max_new_tokens, temperature: (
        _time.sleep(1.0) or "slow done"
    )
    app.state.generator = slow

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(client.post, "/generate", json={"prompt": f"req {i}"})
            for i in range(2)
        ]
        codes = sorted(f.result().status_code for f in futures)

    assert codes == [200, 503]
    busy = next(f.result() for f in futures if f.result().status_code == 503)
    assert busy.json()["error"] == "server_busy"
    assert busy.headers.get("retry-after") == "5"


def test_invalid_env_values_fall_back_to_defaults(client, monkeypatch):
    """Garbage config must degrade gracefully, never crash a request."""
    monkeypatch.setenv("GENERATION_QUEUE_TIMEOUT_S", "not-a-number")
    app.state.generator = StubGenerator(reply="still fine")
    resp = client.post("/generate", json={"prompt": "hello"})
    assert resp.status_code == 200
    assert resp.json()["generated_text"] == "still fine"


def test_acquire_helper_success_timeout_and_no_permit_leak():
    """Unit-test the timeout-safe semaphore acquire used for backpressure."""
    import asyncio

    from app.main import _acquire_with_timeout

    async def scenario():
        sem = asyncio.Semaphore(1)
        # Free semaphore: acquires immediately.
        assert await _acquire_with_timeout(sem, 0.5) is True
        # Held semaphore: times out and reports busy.
        assert await _acquire_with_timeout(sem, 0.05) is False
        # After a timeout, capacity is intact (no leaked permit): release once
        # and the next acquire must succeed.
        sem.release()
        assert await _acquire_with_timeout(sem, 0.05) is True
        sem.release()
        # timeout<=0 waits indefinitely: verify it completes once freed.
        await _acquire_with_timeout(sem, 0)
        release_soon = asyncio.get_event_loop().call_later(0.05, sem.release)
        assert await _acquire_with_timeout(sem, 0) is True
        release_soon.cancel()

    asyncio.run(scenario())


def test_unexpected_exception_returns_consistent_json_without_internals():
    """Non-GenerationError failures get the same clean error shape."""
    with TestClient(app, raise_server_exceptions=False) as test_client:
        stub = StubGenerator()

        def explode(prompt, max_new_tokens, temperature):
            raise RuntimeError("secret internal detail")

        stub.generate = explode
        app.state.generator = stub
        resp = test_client.post("/generate", json={"prompt": "boom"})

    assert resp.status_code == 500
    body = resp.json()
    assert body["error"] == "internal_error"
    assert "secret" not in (body.get("detail") or "")
