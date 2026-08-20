from __future__ import annotations

import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from kairyu_bench.target import Endpoint, PreflightError, TargetClient


class _KairyuHandler(BaseHTTPRequestHandler):
    requests: list[dict[str, Any]] = []

    def log_message(self, format: str, *args: object) -> None:
        pass

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        type(self).requests.append(
            {
                "method": "GET",
                "path": self.path,
                "auth": self.headers.get("Authorization"),
            }
        )
        if self.path == "/v1/models":
            self._json(
                200,
                {
                    "object": "list",
                    "data": [
                        {"id": "embedding-only", "object": "model"},
                        {"id": "chat-capable", "object": "model"},
                    ],
                },
            )
            return
        self._json(404, {"error": "not found"})

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length))
        type(self).requests.append(
            {
                "method": "POST",
                "path": self.path,
                "auth": self.headers.get("Authorization"),
                "payload": payload,
            }
        )
        if self.path == "/v1/chat/completions":
            if payload["model"] == "embedding-only":
                self._json(400, {"error": {"message": "not a chat model"}})
                return
            self._json(
                200,
                {
                    "id": "probe",
                    "object": "chat.completion",
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": "OK"}}
                    ],
                },
            )
            return
        if self.path == "/v1/embeddings":
            if payload["model"] == "embedding-only":
                self._json(
                    200,
                    {
                        "object": "list",
                        "model": "embedding-only",
                        "data": [{"index": 0, "embedding": [0.25, -0.5]}],
                    },
                )
                return
            self._json(400, {"error": {"message": "not an embedding model"}})
            return
        self._json(404, {"error": "not found"})


class TargetClientContractTest(unittest.TestCase):
    def setUp(self) -> None:
        _KairyuHandler.requests = []
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _KairyuHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    @property
    def root(self) -> str:
        host, port = self.server.server_address
        return f"http://{host}:{port}"

    def test_endpoint_accepts_server_root_v1_or_chat_completion_url(self) -> None:
        cases = [
            (
                "https://api.example.test",
                "https://api.example.test/v1",
                "https://api.example.test",
            ),
            (
                "https://api.example.test/v1/",
                "https://api.example.test/v1",
                "https://api.example.test",
            ),
            (
                "https://api.example.test/v1/chat/completions",
                "https://api.example.test/v1",
                "https://api.example.test",
            ),
            (
                "https://api.example.test/prefix/v1",
                "https://api.example.test/prefix/v1",
                "https://api.example.test/prefix",
            ),
        ]

        for supplied, expected, anthropic_base in cases:
            with self.subTest(supplied=supplied):
                endpoint = Endpoint.parse(supplied)
                self.assertEqual(endpoint.base_url, expected)
                self.assertEqual(endpoint.anthropic_base_url, anthropic_base)
                self.assertEqual(endpoint.models_url, f"{expected}/models")
                self.assertEqual(endpoint.chat_url, f"{expected}/chat/completions")

    def test_discovery_uses_first_model_that_accepts_chat_in_server_order(self) -> None:
        client = TargetClient(Endpoint.parse(self.root), api_key="secret", timeout=2)

        model = client.discover_chat_model()

        self.assertEqual(model, "chat-capable")
        self.assertEqual(
            [
                (request["method"], request["path"])
                for request in _KairyuHandler.requests
            ],
            [
                ("GET", "/v1/models"),
                ("POST", "/v1/chat/completions"),
                ("POST", "/v1/chat/completions"),
            ],
        )
        self.assertEqual(
            [request["auth"] for request in _KairyuHandler.requests],
            ["Bearer secret", "Bearer secret", "Bearer secret"],
        )

    def test_discovery_chat_probe_uses_minimal_completion_budget(self) -> None:
        client = TargetClient(Endpoint.parse(self.root), timeout=2)

        client.discover_chat_model()

        chat_probes = [
            request["payload"]
            for request in _KairyuHandler.requests
            if request["path"] == "/v1/chat/completions"
        ]
        self.assertEqual([probe["max_tokens"] for probe in chat_probes], [1, 1])

    def test_embedding_discovery_is_independent_and_authenticates_its_probe(self) -> None:
        client = TargetClient(Endpoint.parse(self.root), api_key="secret", timeout=2)

        model = client.discover_embedding_model()

        self.assertEqual(model, "embedding-only")
        self.assertEqual(
            [(request["method"], request["path"]) for request in _KairyuHandler.requests],
            [("GET", "/v1/models"), ("POST", "/v1/embeddings")],
        )
        self.assertEqual(
            [request["auth"] for request in _KairyuHandler.requests],
            ["Bearer secret", "Bearer secret"],
        )
        self.assertEqual(
            _KairyuHandler.requests[-1]["payload"]["input"],
            ["kairyu-bench embedding capability probe"],
        )

    def test_embedding_discovery_reports_each_rejected_advertised_model(self) -> None:
        class ChatOnlyHandler(_KairyuHandler):
            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length))
                type(self).requests.append(
                    {
                        "method": "POST",
                        "path": self.path,
                        "auth": self.headers.get("Authorization"),
                        "payload": payload,
                    }
                )
                self._json(400, {"error": {"message": "chat only"}})

        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        ChatOnlyHandler.requests = []
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), ChatOnlyHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

        client = TargetClient(Endpoint.parse(self.root), timeout=2)

        with self.assertRaisesRegex(
            PreflightError,
            r"no advertised model accepted embedding requests .*embedding-only: HTTP 400.*chat-capable: HTTP 400",
        ):
            client.discover_embedding_model()

    def test_discovery_fails_when_models_response_has_no_ids(self) -> None:
        class EmptyModelsHandler(_KairyuHandler):
            def do_GET(self) -> None:
                self._json(200, {"object": "list", "data": []})

        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), EmptyModelsHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

        client = TargetClient(Endpoint.parse(self.root), timeout=2)

        with self.assertRaisesRegex(PreflightError, "no model IDs"):
            client.discover_chat_model()

    def test_chat_returns_assistant_text_from_real_api_response(self) -> None:
        client = TargetClient(Endpoint.parse(self.root), api_key="secret", timeout=2)

        response = client.chat(
            "chat-capable",
            [{"role": "user", "content": "Question"}],
            max_tokens=32,
        )

        self.assertEqual(response, "OK")
        request = _KairyuHandler.requests[-1]
        self.assertEqual(request["payload"]["model"], "chat-capable")
        self.assertEqual(request["payload"]["max_tokens"], 32)
        self.assertEqual(
            request["payload"]["messages"],
            [{"role": "user", "content": "Question"}],
        )


if __name__ == "__main__":
    unittest.main()
