#!/usr/bin/env python3
"""
Shared batched inference server for the human-like policy network.

Clients send a fixed-size binary board request over TCP. Requests arriving
within a short interval are evaluated together so one model instance can serve
many concurrently running GTP engines.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
from pathlib import Path
import queue
import socket
import struct
import sys
import threading
import time
from typing import Optional

import numpy as np

from blend_policy_with_egaroucid import (
    BIT_MASKS,
    BLACK,
    HW2,
    INPUT_SIZE,
    POLICY_SIZE,
    BinaryPolicyNetwork,
    default_weights_file,
)


REQUEST_STRUCT = struct.Struct("<QQB")
RESPONSE_SIZE = POLICY_SIZE * 4


def default_model_file() -> Path:
    return default_weights_file().with_name("selected_model.h5")


def recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks = []
    remaining = size
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise EOFError("connection closed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def boards_to_features(requests: list["InferenceRequest"]) -> np.ndarray:
    player = np.empty((len(requests), 1), dtype=np.uint64)
    opponent = np.empty((len(requests), 1), dtype=np.uint64)
    for i, request in enumerate(requests):
        if request.side == BLACK:
            player[i, 0] = request.black
            opponent[i, 0] = request.white
        else:
            player[i, 0] = request.white
            opponent[i, 0] = request.black
    x = np.empty((len(requests), INPUT_SIZE), dtype=np.float32)
    x[:, :HW2] = ((player & BIT_MASKS) != 0).astype(np.float32)
    x[:, HW2:] = ((opponent & BIT_MASKS) != 0).astype(np.float32)
    return x


class PolicyBackend:
    def __init__(self, backend: str, weights: Path, model: Path):
        self.name = backend
        self.device = "CPU"
        self.binary_network: Optional[BinaryPolicyNetwork] = None
        self.tensorflow_infer = None

        if backend in ("auto", "tensorflow"):
            try:
                import tensorflow as tf

                gpus = tf.config.list_physical_devices("GPU")
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                keras_model = tf.keras.models.load_model(model, compile=False)

                @tf.function(
                    input_signature=[tf.TensorSpec(shape=(None, INPUT_SIZE), dtype=tf.float32)],
                    reduce_retracing=True,
                )
                def infer(x):
                    return keras_model(x, training=False)

                self.tensorflow_infer = infer
                self.name = "tensorflow"
                self.device = "GPU" if gpus else "CPU"
                infer(tf.zeros((1, INPUT_SIZE), dtype=tf.float32)).numpy()
                return
            except Exception:
                if backend == "tensorflow":
                    raise

        self.binary_network = BinaryPolicyNetwork(weights)
        self.name = "numpy"
        self.device = "CPU"

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.tensorflow_infer is not None:
            return self.tensorflow_infer(x).numpy().astype(np.float32, copy=False)
        if self.binary_network is None:
            raise RuntimeError("policy backend is not initialized")
        return self.binary_network.predict(x)


@dataclass
class InferenceRequest:
    black: int
    white: int
    side: int
    done: threading.Event = field(default_factory=threading.Event)
    policy: Optional[np.ndarray] = None
    error: Optional[BaseException] = None


class PolicyBatchServer:
    def __init__(
        self,
        host: str,
        port: int,
        backend: PolicyBackend,
        max_batch_size: int,
        batch_wait_ms: float,
        stats_path: Optional[Path],
    ):
        self.host = host
        self.port = int(port)
        self.backend = backend
        self.max_batch_size = int(max_batch_size)
        self.batch_wait_sec = float(batch_wait_ms) / 1000.0
        self.stats_path = stats_path
        self.started_at = time.time()
        self.request_queue: queue.Queue[Optional[InferenceRequest]] = queue.Queue()
        self.shutdown_event = threading.Event()
        self.listener: Optional[socket.socket] = None
        self.connections: set[socket.socket] = set()
        self.connections_lock = threading.Lock()
        self.stats_lock = threading.Lock()
        self.n_requests = 0
        self.n_batches = 0
        self.max_observed_batch_size = 0
        self.total_inference_sec = 0.0
        self.accept_thread: Optional[threading.Thread] = None
        self.worker_thread: Optional[threading.Thread] = None

    def start(self) -> int:
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((self.host, self.port))
        listener.listen()
        listener.settimeout(0.5)
        self.listener = listener
        self.port = int(listener.getsockname()[1])
        self.accept_thread = threading.Thread(target=self._accept_loop, name="policy-accept", daemon=True)
        self.worker_thread = threading.Thread(target=self._batch_loop, name="policy-batch", daemon=True)
        self.accept_thread.start()
        self.worker_thread.start()
        return self.port

    def _accept_loop(self) -> None:
        while not self.shutdown_event.is_set():
            try:
                conn, _ = self.listener.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            conn.settimeout(None)
            with self.connections_lock:
                self.connections.add(conn)
            threading.Thread(target=self._client_loop, args=(conn,), daemon=True).start()

    def _client_loop(self, conn: socket.socket) -> None:
        try:
            while not self.shutdown_event.is_set():
                payload = recv_exact(conn, REQUEST_STRUCT.size)
                black, white, side = REQUEST_STRUCT.unpack(payload)
                request = InferenceRequest(int(black), int(white), int(side))
                self.request_queue.put(request)
                request.done.wait()
                if request.error is not None or request.policy is None:
                    raise RuntimeError("batch inference failed") from request.error
                conn.sendall(request.policy.astype("<f4", copy=False).tobytes())
        except (EOFError, OSError, RuntimeError):
            pass
        finally:
            with self.connections_lock:
                self.connections.discard(conn)
            try:
                conn.close()
            except OSError:
                pass

    def _batch_loop(self) -> None:
        while not self.shutdown_event.is_set():
            try:
                first = self.request_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if first is None:
                break
            requests = [first]
            deadline = time.perf_counter() + self.batch_wait_sec
            while len(requests) < self.max_batch_size:
                remaining = deadline - time.perf_counter()
                if remaining <= 0.0:
                    break
                try:
                    request = self.request_queue.get(timeout=remaining)
                except queue.Empty:
                    break
                if request is None:
                    self.shutdown_event.set()
                    break
                requests.append(request)

            start = time.perf_counter()
            try:
                policies = self.backend.predict(boards_to_features(requests))
                if policies.shape != (len(requests), POLICY_SIZE):
                    raise ValueError(f"unexpected policy output shape: {policies.shape}")
                for request, policy in zip(requests, policies):
                    request.policy = policy
            except BaseException as exc:
                for request in requests:
                    request.error = exc
            finally:
                elapsed = time.perf_counter() - start
                with self.stats_lock:
                    self.n_requests += len(requests)
                    self.n_batches += 1
                    self.max_observed_batch_size = max(self.max_observed_batch_size, len(requests))
                    self.total_inference_sec += elapsed
                for request in requests:
                    request.done.set()

    def stats(self) -> dict:
        with self.stats_lock:
            n_requests = self.n_requests
            n_batches = self.n_batches
            max_batch = self.max_observed_batch_size
            inference_sec = self.total_inference_sec
        return {
            "backend": self.backend.name,
            "device": self.backend.device,
            "host": self.host,
            "port": self.port,
            "max_batch_size": self.max_batch_size,
            "batch_wait_ms": self.batch_wait_sec * 1000.0,
            "requests": n_requests,
            "batches": n_batches,
            "average_batch_size": n_requests / n_batches if n_batches else 0.0,
            "max_observed_batch_size": max_batch,
            "total_inference_sec": inference_sec,
            "average_inference_ms_per_batch": 1000.0 * inference_sec / n_batches if n_batches else 0.0,
            "elapsed_sec": time.time() - self.started_at,
        }

    def close(self) -> None:
        if self.shutdown_event.is_set():
            return
        self.shutdown_event.set()
        self.request_queue.put(None)
        listener = self.listener
        self.listener = None
        if listener is not None:
            try:
                listener.close()
            except OSError:
                pass
        with self.connections_lock:
            connections = list(self.connections)
        for conn in connections:
            try:
                conn.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                conn.close()
            except OSError:
                pass
        if self.worker_thread is not None:
            self.worker_thread.join(timeout=5.0)
        if self.stats_path is not None:
            self.stats_path.parent.mkdir(parents=True, exist_ok=True)
            with self.stats_path.open("w", encoding="utf-8") as f:
                json.dump(self.stats(), f, ensure_ascii=False, indent=2)


def make_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve batched policy-network inference over local TCP.")
    parser.add_argument("--weights", type=Path, default=default_weights_file())
    parser.add_argument("--model", type=Path, default=default_model_file())
    parser.add_argument("--backend", choices=("auto", "tensorflow", "numpy"), default="auto")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--max-batch-size", type=int, default=128)
    parser.add_argument("--batch-wait-ms", type=float, default=2.0)
    parser.add_argument("--stats-path", type=Path, default=None)
    return parser


def main() -> None:
    args = make_argparser().parse_args()
    if args.max_batch_size < 1:
        raise ValueError("--max-batch-size must be positive")
    if args.batch_wait_ms < 0.0:
        raise ValueError("--batch-wait-ms must be non-negative")
    backend = PolicyBackend(args.backend, args.weights, args.model)
    server = PolicyBatchServer(
        args.host,
        args.port,
        backend,
        args.max_batch_size,
        args.batch_wait_ms,
        args.stats_path,
    )
    port = server.start()
    print(f"READY {port} {backend.name} {backend.device}", flush=True)
    try:
        for line in sys.stdin:
            if line.strip().lower() == "quit":
                break
    finally:
        server.close()


if __name__ == "__main__":
    main()
