# Security

Please report a security issue privately through [GitHub security advisories](https://github.com/Vivek0712/strands-decider-agentcore/security/advisories/new) rather than a public issue. Issues in AWS services themselves go to [AWS vulnerability reporting](https://aws.amazon.com/security/vulnerability-reporting/).

## What this sample does and does not do

- **Authorization.** Runtimes use AgentCore's default IAM (SigV4) authorization. Grant `bedrock-agentcore:InvokeAgentRuntime` only to the principals that should call a decider, scoped to the runtime ARN.
- **No credentials in the container.** The image holds no secrets and downloads public, commit-pinned weights from Hugging Face. Pin `DECIDER_REVISION` to a full commit hash so the weights cannot change between cold starts.
- **Non-root.** The container runs as an unprivileged user (uid 1000).
- **Untrusted input.** The `state` you send is text the model reads; a decider can be persuaded by adversarial text like any model. Treat its probabilities as one signal, keep deterministic checks (amounts, permissions) in your own code, and send uncertain or high-impact cases to a person.
- **Local tools.** The triage console binds to 127.0.0.1 and is a demo, not a production service: it has no authentication.
- **Account-level setting.** The first `agentcore deploy` in an account turns on CloudWatch Transaction Search.
