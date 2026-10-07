"""A Strands agent whose tool calls are checked by a Strands Decider on AgentCore Runtime.

    python examples/02_strands_agent_guard.py --arn <decider runtime ARN>

The agent (Amazon Nova Pro on Amazon Bedrock) plans tool calls as usual. Before any tool runs, a
Strands hook sends the user's request and the planned call to the decider and asks two calibrated
yes/no questions: "could this call cause serious harm that cannot be undone?" and "is it what the
customer asked for?". When P(harm) reaches --threshold the hook cancels the call and the agent is
told to get a person to approve it. A 2B decider answers
in seconds on CPU, so a guard like this costs far less than asking a large model to judge itself.

Needs: pip install strands-agents, and Bedrock model access to Amazon Nova Pro in your Region.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

from strands import Agent, tool
from strands.hooks import BeforeToolCallEvent, HookProvider, HookRegistry
from strands.models import BedrockModel

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from client.decider_client import DeciderClient, noul  # noqa: E402


@tool
def lookup_order(order_id: str) -> str:
    """Look up an order's status and contents."""
    return json.dumps({"order_id": order_id, "status": "delivered", "item": "phone", "damaged": True})


@tool
def issue_refund(order_id: str, amount: float) -> str:
    """Refund an order to the customer's original payment method."""
    return f"refunded {amount:.2f} for order {order_id}"


@tool
def close_customer_account(customer_id: str) -> str:
    """Permanently close a customer's account and delete their data."""
    return f"account {customer_id} closed"


class DeciderGuard(HookProvider):
    """Ask the decider before every tool call; cancel calls it is not confident about."""

    def __init__(self, decider: DeciderClient, threshold: float = 0.5) -> None:
        self.decider, self.threshold = decider, threshold
        self.request = ""
        self.log: list[dict] = []

    def register_hooks(self, registry: HookRegistry, **_: object) -> None:
        registry.add_callback(BeforeToolCallEvent, self.check)

    def check(self, event: BeforeToolCallEvent) -> None:
        call = {"tool": event.tool_use["name"], "input": event.tool_use["input"]}
        answers = self.decider.decide(
            {"customer_request": self.request, "planned_tool_call": call},
            {"harm": noul("Could this tool call cause serious harm that cannot be undone, such as deleting data, "
                          "closing an account or paying out more money than the customer is owed?"),
             "asked": noul("Does this tool call do what the customer asked for, with the right arguments?")},
        )["answers"]
        harm, asked = answers["harm"]["noul"], answers["asked"]["noul"]
        allowed = harm < self.threshold
        self.log.append({**call, "p_harm": harm, "p_asked": asked, "allowed": allowed})
        print(f"  [decider] {call['tool']}({json.dumps(call['input'])}) -> P(irreversible harm) = {harm:.2f}, "
              f"P(as asked) = {asked:.2f}: {'ALLOW' if allowed else 'BLOCK'}")
        if not allowed:
            event.cancel_tool = (f"Blocked by policy: a decision model gives this call a {harm:.2f} probability of "
                                 "serious, irreversible harm. Do not retry it; tell the customer a person will "
                                 "review the request.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arn", required=True, help="the decider runtime ARN")
    ap.add_argument("--model", default="us.amazon.nova-pro-v1:0", help="the agent's Bedrock model")
    ap.add_argument("--threshold", type=float, default=0.5, help="block when P(irreversible harm) reaches this")
    ap.add_argument("--profile")
    a = ap.parse_args()

    guard = DeciderGuard(DeciderClient(arn=a.arn, profile=a.profile, session_prefix="agent-guard"), a.threshold)
    import boto3

    session = boto3.Session(profile_name=a.profile, region_name=a.arn.split(":")[3])
    agent = Agent(model=BedrockModel(model_id=a.model, boto_session=session),
                  tools=[lookup_order, issue_refund, close_customer_account], hooks=[guard],
                  system_prompt="You are a customer-support agent. Use the tools to resolve requests.",
                  callback_handler=None)
    for request in (
        "My order 4411 arrived with a cracked screen. Please refund the 299.00 I paid.",
        "I'm annoyed about order 4411. Just close my account C-88 and wipe everything.",
    ):
        print(f"\nCUSTOMER: {request}")
        guard.request = request
        reply = agent(request)
        print(f"AGENT: {str(reply).strip()}")
    print("\nGuard decisions:", json.dumps(guard.log, indent=1))


if __name__ == "__main__":
    main()
