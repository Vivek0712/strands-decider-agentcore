"""deciderctl edits agentcore.json the way the AgentCore CLI expects."""

import json

import pytest

import deciderctl


@pytest.fixture
def project(tmp_path, monkeypatch):
    p = tmp_path / "agentcore.json"
    p.write_text(json.dumps({"name": "StrandsDeciders", "version": 1, "runtimes": [], "memories": []}))
    monkeypatch.setattr(deciderctl, "PROJECT", p)
    monkeypatch.setattr(deciderctl, "STATE", tmp_path / "state.json")
    return p


def test_add_official_release_is_pinned(project):
    assert deciderctl.main(["add", "triage", "--model", "v21"]) == 0
    r = json.loads(project.read_text())["runtimes"][0]
    env = {e["name"]: e["value"] for e in r["envVars"]}
    assert env["DECIDER_MODEL"] == "StrandsAgents/strands-decider-2B-hobson-v21"
    assert env["DECIDER_REVISION"] == deciderctl.CATALOG["v21"][1] and env["DECIDER_QUANT"] == "int8"
    assert r["build"] == "Container" and r["codeLocation"] == "runtime/" and r["protocol"] == "HTTP"
    assert r["lifecycleConfiguration"] == {"idleRuntimeSessionTimeout": 1800, "maxLifetime": 28800}


def test_add_custom_model_and_quant(project, capsys):
    deciderctl.main(["add", "mine", "--model", "my-org/my-decider@" + "c" * 40, "--quant", "bf16",
                     "--max-tokens", "2048"])
    env = {e["name"]: e["value"] for e in json.loads(project.read_text())["runtimes"][0]["envVars"]}
    assert env["DECIDER_MODEL"] == "my-org/my-decider" and env["DECIDER_QUANT"] == "bf16"
    assert env["DECIDER_MAX_TOKENS"] == "2048"
    deciderctl.main(["add", "loose", "--model", "my-org/other"])
    assert "not pinned" in capsys.readouterr().err


@pytest.mark.parametrize("args", [["add", "1bad"], ["add", "ok", "--model", "not a repo"]])
def test_add_refuses_bad_input(project, args):
    with pytest.raises(SystemExit):
        deciderctl.main(args)


def test_duplicate_remove_and_list(project, capsys):
    deciderctl.main(["add", "a", "--model", "v19"])
    with pytest.raises(SystemExit, match="already exists"):
        deciderctl.main(["add", "a", "--model", "v19"])
    deciderctl.main(["list"])
    assert "strands-decider-2B-hobson-v19@bb282d7" in capsys.readouterr().out
    deciderctl.main(["remove", "a"])
    assert json.loads(project.read_text())["runtimes"] == []
    with pytest.raises(SystemExit):
        deciderctl.main(["remove", "a"])
