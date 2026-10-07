"""Call a deployed Strands Decider with plain boto3: one request, three typed questions.

    python examples/01_invoke_boto3.py --arn <runtime ARN from `python deciderctl.py list`>

No helper library: this is the whole wire format. The first call to a new session starts a
microVM and loads the model (about a minute); later calls with the same session id are warm.
"""

import argparse
import json
import time
import uuid

import boto3
from botocore.config import Config

ap = argparse.ArgumentParser()
ap.add_argument("--arn", required=True)
ap.add_argument("--session", default=f"quickstart-{uuid.uuid4()}", help="reuse it to stay on a warm microVM")
ap.add_argument("--profile")
a = ap.parse_args()

client = boto3.Session(profile_name=a.profile, region_name=a.arn.split(":")[3]).client(
    "bedrock-agentcore", config=Config(read_timeout=900))

request = {
    "state": "Order #4411 arrived with a cracked screen. I want my money back today, this is the second time.",
    "questions": {
        "escalate": {"type": "noul", "instructions": "Should this ticket go to a human agent?"},
        "queue": {"type": "choice", "instructions": "Which queue handles this ticket?",
                  "criteria": {"returns": "refunds, returns and damaged items", "billing": "payments and charges",
                               "shipping": "late or lost deliveries", "account": "login and profile"}},
        "urgency": {"type": "score", "instructions": "How urgent is this ticket?",
                    "criteria": ["low: can wait a week", "medium: within two days", "high: today"]},
    },
}

for attempt in ("first call (cold if the session is new)", "second call (same session: warm)"):
    t = time.time()
    resp = client.invoke_agent_runtime(agentRuntimeArn=a.arn, runtimeSessionId=a.session,
                                       payload=json.dumps(request).encode(), contentType="application/json")
    answer = json.loads(resp["response"].read())
    print(f"\n{attempt}: {time.time() - t:.1f}s")
    print(json.dumps(answer, indent=2))
