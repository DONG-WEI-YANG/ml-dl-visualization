import asyncio
import threading

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.api import model_routes as routes
from app.models.linear import run_gradient_descent


@pytest.mark.parametrize('model,payload', [
    (routes.GradientDescentRequest, {'X': [[1]], 'y': [2], 'epochs': 0}),
    (routes.GradientDescentRequest, {'X': [[1]], 'y': [2], 'epochs': 1001}),
    (routes.GradientDescentRequest, {'X': [], 'y': []}),
    (routes.GradientDescentRequest, {'X': [[1], [1, 2]], 'y': [2, 3]}),
    (routes.GradientDescentRequest, {'X': [[1]], 'y': []}),
    (routes.GradientDescentRequest, {'X': [[float('nan')]], 'y': [2]}),
    (routes.GradientDescentRequest, {'X': [[1]] * 501, 'y': [2] * 501}),
    (routes.GradientDescentRequest, {'X': [[1] * 21], 'y': [2]}),
    (routes.LossLandscapeRequest, {'resolution': 100000}),
    (routes.LossLandscapeRequest, {'w0_range': [1]}),
    (routes.LossLandscapeRequest, {'w0_range': [2, 1]}),
    (routes.LossLandscapeRequest, {'surface_type': 'unknown'}),
    (routes.DecisionBoundaryRequest, {'X': [[1], [2]], 'y': [0, 1]}),
    (routes.DecisionBoundaryRequest, {'X': [[1, 2], [2, 3]], 'y': [0, 0]}),
    (routes.DecisionBoundaryRequest, {'X': [[1, 2], [2, 3]], 'y': [0, 1], 'model_type': 'bad'}),
    (routes.RocPrRequest, {'X': [[1], [2]], 'y': [2, 3]}),
    (routes.TreeModelRequest, {'X': [[1], [2]], 'y': [0, 1], 'n_estimators': 201}),
    (routes.TreeModelRequest, {'X': [[1], [2]], 'y': [0, 1], 'max_depth': 16}),
    (routes.TreeModelRequest, {'X': [[1], [2]], 'y': [0, 1], 'feature_names': ['a', 'b']}),
])
def test_reject_invalid_requests(model, payload):
    with pytest.raises(ValidationError):
        model(**payload)


def test_final_loss_matches_returned_weights():
    result = run_gradient_descent([[1]], [2], learning_rate=0.1, epochs=1)
    w, b = result.final_weights
    assert result.final_loss == pytest.approx((w + b - 2) ** 2)
    assert result.loss_history == [4.0]  # History records pre-update states.


def test_invalid_and_divergent_requests_return_422():
    app = FastAPI()
    app.include_router(routes.router)
    with TestClient(app, raise_server_exceptions=False) as client:
        for payload in [
            {'X': [[1]], 'y': [2], 'epochs': 0},
            {'X': [[1000]], 'y': [1000], 'learning_rate': 1, 'epochs': 1000},
        ]:
            assert client.post('/api/models/gradient-descent', json=payload).status_code == 422


def test_compute_runs_off_event_loop_and_rejects_overload(monkeypatch):
    entered = threading.Event()
    release = threading.Event()
    threads = []
    def compute(*args):
        threads.append(threading.get_ident())
        entered.set()
        assert release.wait(5)
        return {'ok': True}
    monkeypatch.setattr(routes, 'run_gradient_descent', compute)

    async def scenario():
        req = routes.GradientDescentRequest(X=[[1]], y=[2])
        jobs = [asyncio.create_task(routes.gradient_descent(req)) for _ in range(2)]
        try:
            for _ in range(100):
                if len(threads) == 2:
                    break
                await asyncio.sleep(.01)
            assert len(threads) == 2
            assert threading.get_ident() not in threads
            with pytest.raises(Exception) as exc:
                await routes.gradient_descent(req)
            assert exc.value.status_code == 503
        finally:
            release.set()
            await asyncio.gather(*jobs)
        assert await routes.gradient_descent(req) == {'ok': True}
    asyncio.run(scenario())


def test_worker_slot_is_released_after_computation_error(monkeypatch):
    def invalid(*args):
        raise ValueError('bad input')
    monkeypatch.setattr(routes, 'run_gradient_descent', invalid)
    app = FastAPI()
    app.include_router(routes.router)
    with TestClient(app) as client:
        for _ in range(4):
            assert client.post('/api/models/gradient-descent', json={'X': [[1]], 'y': [2]}).status_code == 422


@pytest.mark.parametrize('path,payload', [
    ('gradient-descent', {'X': [[1], [2]], 'y': [2, 4]}),
    ('loss-landscape', {'resolution': 10}),
    ('decision-boundary', {'X': [[0, 0], [1, 1]], 'y': [0, 1]}),
    ('roc-pr', {'X': [[0, 0], [1, 1]], 'y': [0, 1]}),
    ('tree', {'X': [[0], [1], [2]], 'y': [0, 1, 2], 'max_depth': 15}),
])
def test_valid_model_requests_keep_working(path, payload):
    app = FastAPI()
    app.include_router(routes.router)
    with TestClient(app) as client:
        assert client.post('/api/models/' + path, json=payload).status_code == 200


def test_cancelled_before_dispatch_does_not_leak_slot(monkeypatch):
    import anyio
    slots = threading.BoundedSemaphore(2)
    monkeypatch.setattr(routes, '_compute_slots', slots)
    async def scenario():
        with anyio.CancelScope() as scope:
            scope.cancel()
            await routes._compute(lambda: None)
        assert slots.acquire(blocking=False)
        assert slots.acquire(blocking=False)
    anyio.run(scenario)


def test_native_cancellation_while_waiting_for_thread_capacity(monkeypatch):
    import anyio
    slots = threading.BoundedSemaphore(2)
    monkeypatch.setattr(routes, '_compute_slots', slots)
    async def scenario():
        limiter = anyio.to_thread.current_default_thread_limiter()
        old = limiter.total_tokens
        limiter.total_tokens = 1
        await limiter.acquire()
        try:
            task = asyncio.create_task(routes._compute(lambda: None))
            await asyncio.sleep(.02)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert slots.acquire(blocking=False)
            assert slots.acquire(blocking=False)
        finally:
            limiter.release()
            limiter.total_tokens = old
    anyio.run(scenario)


def test_native_cancellation_keeps_running_worker_slot(monkeypatch):
    import anyio
    slots = threading.BoundedSemaphore(1)
    monkeypatch.setattr(routes, '_compute_slots', slots)
    started, finish, done = threading.Event(), threading.Event(), threading.Event()
    def worker():
        started.set()
        try:
            assert finish.wait(5)
        finally:
            done.set()
    async def scenario():
        task = asyncio.create_task(routes._compute(worker))
        try:
            for _ in range(100):
                if started.is_set(): break
                await asyncio.sleep(.01)
            assert started.is_set()
            task.cancel()
            with pytest.raises(asyncio.CancelledError): await task
            assert not slots.acquire(blocking=False)
        finally:
            finish.set()
        for _ in range(100):
            if slots.acquire(blocking=False): return
            await asyncio.sleep(.01)
        pytest.fail('Worker failed to return slot')
    anyio.run(scenario)
