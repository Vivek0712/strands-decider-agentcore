"""AgentCore Runtime entrypoint: a Strands Decider model behind /invocations and /ping."""

from bedrock_agentcore.runtime import BedrockAgentCoreApp
from bedrock_agentcore.runtime.models import PingStatus

from decider_runtime.config import Settings
from decider_runtime.engine import DeciderEngine
from decider_runtime.protocol import handle

settings = Settings.from_env()
engine = DeciderEngine(settings)
app = BedrockAgentCoreApp()


@app.ping
def ping() -> PingStatus:
    # HealthyBusy while the model loads tells AgentCore the session is working, not idle
    return PingStatus.HEALTHY if engine.ready.is_set() else PingStatus.HEALTHY_BUSY


@app.entrypoint
def invoke(payload, context=None):
    return handle(payload, engine, settings)


if __name__ == "__main__":
    if settings.sync_load:
        # Lambda (behind the Lambda Web Adapter) and SageMaker send traffic as soon as /ping answers,
        # so load first and start serving only when the model is ready
        engine.load()
    else:
        engine.start_background_load()  # AgentCore: /ping answers at once, the model loads behind it
    app.run()
