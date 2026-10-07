#!/usr/bin/env python3
"""deciderctl: add Strands Decider models to this AgentCore project, then deploy them with the
AgentCore CLI.

    python deciderctl.py target                         # aws-targets.json from your current credentials
    python deciderctl.py add triage --model v21         # an official release, pinned to its commit
    python deciderctl.py add legacy --model v19 --quant bf16
    python deciderctl.py add mine --model my-org/my-decider@<commit>
    python deciderctl.py list
    agentcore deploy -y                                 # the AgentCore CLI builds and deploys everything
    python deciderctl.py health triage                  # starts a session and reports load timings
    python deciderctl.py ask triage --state "Order arrived broken" --noul "Should a human handle this?"
    python deciderctl.py remove legacy && agentcore deploy -y

Every decider is one AgentCore Runtime built from the same container (runtime/), configured
by environment variables: which model, which revision, int8 or bf16. Standard library only.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent
PROJECT = ROOT / "agentcore" / "agentcore.json"
TARGETS = ROOT / "agentcore" / "aws-targets.json"
STATE = ROOT / "agentcore" / ".cli" / "deployed-state.json"

# Official Strands Decider releases, pinned to the exact Hugging Face commit.
CATALOG = {
    "v21": ("StrandsAgents/strands-decider-2B-hobson-v21", "2b52a6235c1b8306bbfa30b00b9d4b74b63a39f5"),
    "v19": ("StrandsAgents/strands-decider-2B-hobson-v19", "bb282d786bc251fd4e3068de3ada9ddbb38127cd"),
}
NAME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9_]{0,47}$")
FULL_SHA = re.compile(r"^[0-9a-f]{40}$")
TAG_KEY = "strands-decider:model"


def load_project() -> dict:
    if not PROJECT.exists():
        sys.exit(f"{PROJECT} not found: run deciderctl from the repository root")
    return json.loads(PROJECT.read_text())


def save_project(p: dict) -> None:
    PROJECT.write_text(json.dumps(p, indent=2) + "\n")


def resolve_model(spec: str) -> tuple[str, str | None]:
    """'v21' -> pinned official release; 'org/repo@commit' -> as given; a directory -> as is."""
    if spec in CATALOG:
        return CATALOG[spec]
    repo, _, rev = spec.partition("@")
    if not re.match(r"^[\w.-]+/[\w.-]+$", repo) and not os.path.isdir(repo):
        sys.exit(f"--model must be one of {sorted(CATALOG)}, org/repo[@commit], or a directory; got {spec!r}")
    if rev and not FULL_SHA.match(rev):
        print(f"warning: {rev!r} is not a full commit hash; the deployed weights can change under you",
              file=sys.stderr)
    if not rev and not os.path.isdir(repo):
        print(f"warning: {repo} is not pinned (use {repo}@<commit>); the latest revision is used at each cold start",
              file=sys.stderr)
    return repo, rev or None


def deciders(p: dict) -> list[dict]:
    return [r for r in p.get("runtimes", []) if (r.get("tags") or {}).get(TAG_KEY)]


def env_of(r: dict) -> dict[str, str]:
    return {e["name"]: e["value"] for e in r.get("envVars", [])}


# ---- commands ------------------------------------------------------------------------------------

def cmd_target(a: argparse.Namespace) -> int:
    import boto3  # only this command and the invoke commands need boto3

    s = boto3.Session(profile_name=a.profile, region_name=a.region)
    account = s.client("sts").get_caller_identity()["Account"]
    region = a.region or s.region_name or "us-east-1"
    TARGETS.write_text(json.dumps([{"name": "default", "account": account, "region": region}], indent=2) + "\n")
    print(f"agentcore/aws-targets.json: account {account}, region {region}")
    return 0


def cmd_add(a: argparse.Namespace) -> int:
    if not NAME_RE.match(a.name):
        sys.exit("name: a letter, then letters, digits or _ (at most 48)")
    p = load_project()
    if any(r["name"] == a.name for r in p.get("runtimes", [])):
        sys.exit(f"{a.name} already exists (deciderctl remove {a.name} first)")
    repo, rev = resolve_model(a.model)
    env = {"DECIDER_MODEL": repo, "DECIDER_QUANT": a.quant, "DECIDER_MAX_TOKENS": str(a.max_tokens),
           "DECIDER_MAX_QUESTIONS": str(a.max_questions)}
    if rev:
        env["DECIDER_REVISION"] = rev
    if a.model_name:
        env["DECIDER_MODEL_NAME"] = a.model_name
    runtime = {
        "name": a.name,
        "description": f"Strands Decider {repo}{'@' + rev[:7] if rev else ''} ({a.quant}) on AgentCore Runtime",
        "build": "Container",
        "entrypoint": "main.py",
        "codeLocation": "runtime/",
        "runtimeVersion": "PYTHON_3_12",
        "networkMode": "PUBLIC",
        "protocol": "HTTP",
        "envVars": [{"name": k, "value": v} for k, v in env.items()],
        # a session keeps its warm model until it is idle this long; loading again costs a cold start
        "lifecycleConfiguration": {"idleRuntimeSessionTimeout": a.idle_timeout, "maxLifetime": a.max_lifetime},
        "tags": {TAG_KEY: repo.split("/")[-1][:256], "strands-decider:quant": a.quant},
    }
    p.setdefault("runtimes", []).append(runtime)
    save_project(p)
    print(f"added {a.name}: {repo}{'@' + rev if rev else ''} ({a.quant}). Deploy with: agentcore deploy -y")
    return 0


def cmd_remove(a: argparse.Namespace) -> int:
    p = load_project()
    before = len(p.get("runtimes", []))
    p["runtimes"] = [r for r in p.get("runtimes", []) if r["name"] != a.name]
    if len(p["runtimes"]) == before:
        sys.exit(f"no decider named {a.name}")
    save_project(p)
    print(f"removed {a.name} from the project. Apply with: agentcore deploy -y")
    return 0


def deployed() -> dict[str, dict]:
    if not STATE.exists():
        return {}
    state = json.loads(STATE.read_text())
    out: dict[str, dict] = {}
    for target in state.get("targets", {}).values():
        out.update(target.get("resources", {}).get("runtimes", {}))
    return out


def cmd_list(a: argparse.Namespace) -> int:
    p, dep = load_project(), deployed()
    rows = deciders(p)
    if not rows:
        print("no deciders yet: python deciderctl.py add <name> --model v21")
        return 0
    print(f"{'name':<16} {'model':<46} {'quant':<6} {'idle':>6}  deployed")
    for r in rows:
        e = env_of(r)
        model = e.get("DECIDER_MODEL", "?") + ("@" + e["DECIDER_REVISION"][:7] if e.get("DECIDER_REVISION") else "")
        idle = (r.get("lifecycleConfiguration") or {}).get("idleRuntimeSessionTimeout", "")
        print(f"{r['name']:<16} {model:<46} {e.get('DECIDER_QUANT', ''):<6} {idle:>6}  "
              f"{dep.get(r['name'], {}).get('runtimeArn', 'no')}")
    return 0


def client_for(name: str, a: argparse.Namespace):  # noqa: ANN201
    sys.path.insert(0, str(ROOT))
    from client.decider_client import DeciderClient

    arn = a.arn or deployed().get(name, {}).get("runtimeArn")
    if not arn:
        sys.exit(f"{name} is not deployed yet (agentcore deploy -y), or pass --arn")
    return DeciderClient(arn=arn, profile=a.profile, region=a.region)


def session_id(name: str, a: argparse.Namespace) -> str:
    """A stable session id, so repeated commands reach the same warm microVM (ids need 33+ characters)."""
    return f"{a.session or 'deciderctl'}-{name}".ljust(33, "0")


def cmd_health(a: argparse.Namespace) -> int:
    print(json.dumps(client_for(a.name, a).health(session_id=session_id(a.name, a)), indent=2))
    return 0


def cmd_ask(a: argparse.Namespace) -> int:
    from client.decider_client import choice, noul, score

    questions: dict = {}
    for i, q in enumerate(a.noul or []):
        questions[f"noul_{i + 1}" if len(a.noul) > 1 else "answer"] = noul(q)
    for i, spec in enumerate(a.choice or []):
        text, _, opts = spec.partition("::")
        questions[f"choice_{i + 1}"] = choice(text, {o.strip(): o.strip() for o in opts.split("|") if o.strip()})
    for i, spec in enumerate(a.score or []):
        text, _, levels = spec.partition("::")
        questions[f"score_{i + 1}"] = score(text, [x.strip() for x in levels.split("|") if x.strip()])
    if a.file:
        payload = json.loads(pathlib.Path(a.file).read_text())
    elif questions and a.state:
        payload = {"state": a.state, "questions": questions}
    else:
        sys.exit("give --state with --noul/--choice/--score, or --file request.json")
    print(json.dumps(client_for(a.name, a).invoke(payload, session_id=session_id(a.name, a)), indent=2))
    return 0


def cmd_deploy(a: argparse.Namespace) -> int:
    """Convenience: run the AgentCore CLI's deploy from the project directory."""
    exe = os.environ.get("AGENTCORE_CLI", "agentcore")
    return subprocess.call([exe, "deploy", "-y", *(["--verbose"] if a.verbose else [])], cwd=ROOT)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="deciderctl", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("target", help="write agentcore/aws-targets.json from your AWS credentials")
    s.add_argument("--profile")
    s.add_argument("--region")
    s.set_defaults(fn=cmd_target)

    s = sub.add_parser("add", help="add a decider runtime to agentcore/agentcore.json")
    s.add_argument("name")
    s.add_argument("--model", default="v21", help=f"{', '.join(sorted(CATALOG))}, org/repo@commit, or a directory")
    s.add_argument("--quant", choices=["int8", "bf16"], default="int8",
                   help="int8 (default): ~4 GB, fastest on the 2 vCPU microVM; bf16: reference precision, slower")
    s.add_argument("--max-tokens", type=int, default=4096, help="prompt window; longer inputs are shortened")
    s.add_argument("--max-questions", type=int, default=16, help="questions per request")
    s.add_argument("--idle-timeout", type=int, default=1800, help="seconds a warm session waits before it stops")
    s.add_argument("--max-lifetime", type=int, default=28800, help="seconds a session lives at most")
    s.add_argument("--model-name", help="name reported in responses (default: repo@commit)")
    s.set_defaults(fn=cmd_add)

    s = sub.add_parser("remove", help="remove a decider runtime")
    s.add_argument("name")
    s.set_defaults(fn=cmd_remove)

    s = sub.add_parser("list", help="deciders in the project and their deployed ARNs")
    s.set_defaults(fn=cmd_list)

    for name, fn, hlp in (("health", cmd_health, "start a session and report model, timings, memory"),
                          ("ask", cmd_ask, "ask a deployed decider a question")):
        s = sub.add_parser(name, help=hlp)
        s.add_argument("name")
        s.add_argument("--arn")
        s.add_argument("--profile")
        s.add_argument("--region")
        s.add_argument("--session", help="session name (default deciderctl): same name, same warm microVM")
        if name == "ask":
            s.add_argument("--state", help="the text (or JSON) to decide about")
            s.add_argument("--noul", action="append", help="a yes/no question (repeatable)")
            s.add_argument("--choice", action="append", help='"question::option a|option b|option c"')
            s.add_argument("--score", action="append", help='"question::low|medium|high"')
            s.add_argument("--file", help="a full request JSON instead")
        s.set_defaults(fn=fn)

    s = sub.add_parser("deploy", help="run `agentcore deploy -y` (AGENTCORE_CLI overrides the binary)")
    s.add_argument("--verbose", action="store_true")
    s.set_defaults(fn=cmd_deploy)

    a = ap.parse_args(argv)
    return int(a.fn(a) or 0)


if __name__ == "__main__":
    sys.exit(main())
