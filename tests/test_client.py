"""DeciderClient against a fake local runtime (the same HTTP contract AgentCore uses)."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from client.decider_client import DeciderClient, DeciderError, choice, noul, score


@pytest.fixture
def fake_runtime():
    seen = []
    state = {"loading_left": 0}

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append((self.headers.get("X-Amzn-Bedrock-AgentCore-Runtime-Session-Id"), body))
            if state["loading_left"] > 0:
                state["loading_left"] -= 1
                out = {"error": {"code": "loading", "message": "still loading"}}
            elif body.get("state") == "bad":
                out = {"error": {"code": "bad_request", "message": "nope"}}
            elif "batch" in body:
                out = {"results": [{"answers": {}} for _ in body["batch"]]}
            else:
                out = {"answers": {k: {"type": "noul", "noul": 0.8} for k in body.get("questions", {})}}
            data = json.dumps(out).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", seen, state
    srv.shutdown()


def test_question_helpers():
    assert noul("Is it?") == {"type": "noul", "instructions": "Is it?"}
    assert choice("Which?", {"a": "A"})["criteria"] == {"a": "A"}
    assert score("How?", ["low", "high"])["criteria"] == ["low", "high"]


def test_decide_batch_and_session_pool(fake_runtime):
    url, seen, _ = fake_runtime
    c = DeciderClient(url=url, pool=2)
    assert c.decide("x", {"q": noul("Is it?")})["answers"]["q"]["noul"] == 0.8
    c.decide("y", {"q": noul("Is it?")})
    assert len(c.decide_batch([{"state": "a", "questions": {}}, {"state": "b", "questions": {}}])) == 2
    sids = [s for s, _ in seen]
    assert len(set(sids)) == 2 and all(len(s) >= 33 for s in sids), "two warm sessions, round robin"


def test_loading_is_retried_and_errors_raise(fake_runtime, monkeypatch):
    url, _, state = fake_runtime
    monkeypatch.setattr("client.decider_client.time.sleep", lambda s: None)
    state["loading_left"] = 2
    c = DeciderClient(url=url)
    assert c.decide("x", {"q": noul("?")})["answers"]["q"]["noul"] == 0.8
    with pytest.raises(DeciderError, match="bad_request"):
        c.decide("bad", {"q": noul("?")})


def test_needs_exactly_one_target():
    with pytest.raises(ValueError):
        DeciderClient()
