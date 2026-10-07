# Ticket-triage console

A small web app (FastAPI and one HTML page) that triages support tickets with a Strands Decider on AgentCore Runtime. Each ticket gets **one invocation with three questions**:

| question | type | used for |
|---|---|---|
| Does this ticket need a human agent? | `noul` with true/false criteria | automate (P ≤ 0.25), assign to a person (P ≥ 0.75), or review |
| Which team should handle it? | `choice` over 6 teams | the queue, if its probability is at least 0.6 |
| How urgent is it? | `score` on a 3-level rubric | the reply-time target (most likely level) |

A ticket is routed without review only when every answer clears its threshold. The decider is calibrated, so the thresholds are a direct trade between how much is automated and how often a person has to look. Tune them on your own tickets (`ESCALATE_YES`, `ESCALATE_NO` and `QUEUE_MIN` in `app.py`).

```bash
python examples/04_triage_console/app.py --arn <triage runtime ARN>    # or --url http://localhost:8080
open http://127.0.0.1:8501                                            # ?ticket=T-1047&run=1 deep-links one ticket
```

The first triage after start-up starts a new AgentCore session (about a minute). After that, each ticket takes 15 to 30 s on the 2 vCPU microVM.

![console](../../docs/img/console-T-1047.png)
