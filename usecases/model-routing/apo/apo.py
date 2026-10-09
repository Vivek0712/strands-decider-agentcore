"""Bedrock Advanced Prompt Optimization (APO) for model selection: one prompt template per task family,
optimized and scored on 5 target models, graded by our own Lambda evaluator.

    python apo/apo.py setup      # S3 bucket, evaluator Lambda (+ permission for Bedrock to invoke it)
    python apo/apo.py input      # data/apo_input.jsonl from the train split, uploaded to S3
    python apo/apo.py create     # starts the job, prints its ARN (saved to results/apo_job.json)
    python apo/apo.py status     # job status
    python apo/apo.py fetch      # downloads the results to results/apo_raw/
    python apo/apo.py parse      # writes results/apo_prompts.json and results/apo_scores.json

Why a Lambda evaluator: every task family here has an exact answer (a number, an option letter, JSON
fields, unit tests), so the same graders that score the benchmark also steer the optimizer. The
built-in judge and custom LLM judges run on Anthropic models, which this account has not enabled.
"""

from __future__ import annotations

import io
import json
import pathlib
import sys
import time
import zipfile

import boto3

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bench"))
from models import MODELS, PROFILE, REGION  # noqa: E402

APO_MODELS = ["micro", "lite", "pro", "haiku45", "sonnet46"]
JUDGE = "anthropic.claude-sonnet-4-6"  # one of the judge models APO allows; used for the code family only
CODE_JUDGE_PROMPT = (
    "You grade a Python answer to a programming task. The reference contains a working solution, then a line "
    "'# tests', then the unit tests the answer's function must pass.\n\nTask and answer format: {{prompt}}\n\n"
    "Candidate answer: {{response}}\n\nReference solution and tests: {{referenceResponse}}\n\n"
    "Trace the candidate's function on every test input. Score 5 if it would pass every test and follows the "
    "requested format (one python code block, no tests, no explanation); 4 if it passes every test but breaks the "
    "format; 3 if it passes most tests; 2 if it passes some; 1 if it passes none or would not run. Explain briefly, "
    "then give the score."
)
FUNCTION = "decider-routing-apo-evaluator"
STATE = ROOT / "results" / "apo_job.json"
session = boto3.Session(profile_name=PROFILE, region_name=REGION)
ACCOUNT = session.client("sts").get_caller_identity()["Account"]
BUCKET = f"decider-routing-apo-{ACCOUNT}-{REGION}"


def lambda_source() -> str:
    return (ROOT / "apo" / "lambda_function.py").read_text()


def setup() -> None:
    s3, iam, lam = session.client("s3"), session.client("iam"), session.client("lambda")
    try:
        s3.create_bucket(Bucket=BUCKET)
        s3.put_public_access_block(
            Bucket=BUCKET,
            PublicAccessBlockConfiguration={
                "BlockPublicAcls": True,
                "IgnorePublicAcls": True,
                "BlockPublicPolicy": True,
                "RestrictPublicBuckets": True,
            },
        )
        print("bucket", BUCKET)
    except s3.exceptions.BucketAlreadyOwnedByYou:
        print("bucket exists", BUCKET)
    role = f"{FUNCTION}-role"
    try:
        arn = iam.create_role(
            RoleName=role,
            AssumeRolePolicyDocument=json.dumps(
                {
                    "Version": "2012-10-17",
                    "Statement": [
                        {
                            "Effect": "Allow",
                            "Principal": {"Service": "lambda.amazonaws.com"},
                            "Action": "sts:AssumeRole",
                        }
                    ],
                }
            ),
        )["Role"]["Arn"]
        iam.attach_role_policy(
            RoleName=role, PolicyArn="arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
        )
        time.sleep(12)  # IAM propagation
    except iam.exceptions.EntityAlreadyExistsException:
        arn = iam.get_role(RoleName=role)["Role"]["Arn"]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("lambda_function.py", lambda_source())
    try:
        lam.create_function(
            FunctionName=FUNCTION,
            Runtime="python3.12",
            Role=arn,
            Handler="lambda_function.lambda_handler",
            Code={"ZipFile": buf.getvalue()},
            Timeout=900,
            MemorySize=1024,
            Architectures=["arm64"],
            Description="Exact graders for the decider model-routing APO job",
        )
    except lam.exceptions.ResourceConflictException:
        lam.update_function_code(FunctionName=FUNCTION, ZipFile=buf.getvalue())
    lam.get_waiter("function_active_v2").wait(FunctionName=FUNCTION)
    try:
        lam.add_permission(
            FunctionName=FUNCTION,
            StatementId="bedrock-apo",
            Action="lambda:InvokeFunction",
            Principal="bedrock.amazonaws.com",
            SourceAccount=ACCOUNT,
        )
    except lam.exceptions.ResourceConflictException:
        pass
    print("lambda", FUNCTION)


def lambda_arn() -> str:
    return session.client("lambda").get_function(FunctionName=FUNCTION)["Configuration"]["FunctionArn"]


def reference_for(task: dict) -> str:
    if task["family"] == "code":
        spec = json.loads(task["answer"])
        return task["reference"].strip() + "\n# tests\n" + "\n".join(spec["imports"] + spec["tests"])
    return task["answer"]


def make_input() -> None:
    tasks = [json.loads(x) for x in (ROOT / "data" / "tasks.jsonl").read_text().splitlines() if x]
    templates = json.loads((ROOT / "data" / "templates.json").read_text())
    arn = lambda_arn()
    lines = []
    for fam, tpl in templates.items():
        samples = [
            {"inputVariables": [{"input": t["input"]}], "referenceResponse": reference_for(t)}
            for t in tasks
            if t["family"] == fam and t["split"] == "train"
        ]
        line = {"version": "bedrock-2026-05-14", "templateId": fam, "promptTemplate": tpl, "evaluationSamples": samples}
        if fam == "code":  # an APO evaluator Lambda may not execute code, so a judge traces the tests instead
            line.update(
                customEvaluationMetricLabel="codetestjudge",
                customLLMJConfig={"customLLMJPrompt": CODE_JUDGE_PROMPT, "customLLMJModelId": JUDGE},
            )
        else:
            line.update(customEvaluationMetricLabel=f"{fam}exactscore", evaluationMetricLambdaArn=arn)
        lines.append(line)
    path = ROOT / "data" / "apo_input.jsonl"
    path.write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    session.client("s3").upload_file(str(path), BUCKET, "input/apo_input.jsonl")
    print(f"{len(lines)} templates, {sum(len(x['evaluationSamples']) for x in lines)} samples -> s3://{BUCKET}/input/")


JOBS = ROOT / "results" / "apo_jobs.json"


def create(models: list[str] | None = None, families: list[str] | None = None) -> None:
    """Start one job for the given target models (default: all of APO_MODELS). Smaller jobs make fewer
    concurrent calls, which matters when the account's Claude quota is small."""
    models = models or APO_MODELS
    br = session.client("bedrock")
    key = "input/apo_input.jsonl"
    if families:  # a retry of only some templates: upload a subset of the input file
        lines = [
            x
            for x in (ROOT / "data" / "apo_input.jsonl").read_text().splitlines()
            if x and json.loads(x)["templateId"] in families
        ]
        key = f"input/apo_input-{'-'.join(families)}.jsonl"
        session.client("s3").put_object(Bucket=BUCKET, Key=key, Body=("\n".join(lines) + "\n").encode())
    name = f"decider-routing-{'-'.join(models)}-{time.strftime('%Y%m%d-%H%M%S')}"
    arn = br.create_advanced_prompt_optimization_job(
        jobName=name,
        jobDescription="Model selection for 6 task families (decider routing benchmark)",
        modelConfigurations=[
            {"modelId": MODELS[m].model_id, "inferenceConfig": {"temperature": 0.0, "maxTokens": 1500}} for m in models
        ],
        inputConfig={"s3Uri": f"s3://{BUCKET}/{key}"},
        outputConfig={"s3Uri": f"s3://{BUCKET}/output/"},
    )["jobArn"]
    STATE.write_text(json.dumps({"jobArn": arn, "jobName": name, "created": time.time()}, indent=2))
    jobs = json.loads(JOBS.read_text()) if JOBS.exists() else []
    JOBS.write_text(json.dumps(jobs + [{"jobArn": arn, "models": models}], indent=2))
    print(arn)


def status() -> dict:
    j = session.client("bedrock").get_advanced_prompt_optimization_job(
        jobIdentifier=json.loads(STATE.read_text())["jobArn"]
    )
    print(json.dumps({k: v for k, v in j.items() if k != "ResponseMetadata"}, indent=2, default=str)[:3000])
    return j


def fetch() -> None:
    j = status()
    s3 = session.client("s3")
    prefix = j["outputConfig"]["s3Uri"].replace(f"s3://{BUCKET}/", "")
    keys = [o["Key"] for o in s3.list_objects_v2(Bucket=BUCKET, Prefix=prefix).get("Contents", [])]
    raw = ROOT / "results" / "apo_raw"
    raw.mkdir(exist_ok=True)
    for k in keys:
        s3.download_file(BUCKET, k, str(raw / k.replace("/", "__")))
    print("downloaded", len(keys), "files to", raw)


def parse() -> None:
    """Turn the job's results file into the two inputs bench/evaluate.py reads:
    results/apo_prompts.json {family: {model: optimized template}} and
    results/apo_scores.json  {family: {model: average score of the optimized prompt (train samples)}}."""
    by_id = {v.model_id: k for k, v in MODELS.items()}
    jobs = [j["jobArn"].split("/")[-1] for j in json.loads(JOBS.read_text())]
    paths = [p for j in jobs for p in (ROOT / "results" / "apo_raw").glob(f"*{j}*results.jsonl")]  # oldest first
    prompts: dict = {}
    scores: dict = {}
    metrics: dict = {}
    failed = []
    for line in [line for path in paths for line in path.read_text().splitlines()]:
        rec = json.loads(line)
        fam = rec["promptTemplateId"]
        for r in rec["promptOptimizationResults"]:
            m = by_id.get(r["modelId"], r["modelId"])
            if r["status"] != "COMPLETED" and not r.get("optimizedPromptMetrics", {}).get("averageScore"):
                failed.append((fam, m, r.get("failureReason", "")[:120]))
                continue
            prompts.setdefault(fam, {})[m] = r["optimizedPromptTemplate"]
            scores.setdefault(fam, {})[m] = r["optimizedPromptMetrics"]["averageScore"]
            metrics.setdefault(fam, {})[m] = {
                "original": r["originalPromptMetrics"],
                "optimized": r["optimizedPromptMetrics"],
            }
    (ROOT / "results" / "apo_prompts.json").write_text(json.dumps(prompts, indent=2))
    (ROOT / "results" / "apo_scores.json").write_text(json.dumps(scores, indent=2))
    (ROOT / "results" / "apo_metrics.json").write_text(json.dumps(metrics, indent=2))
    failed = [f for f in failed if f[1] not in scores.get(f[0], {})]  # a later job may have succeeded
    print(f"{sum(len(v) for v in scores.values())} optimized (family, model) pairs; {len(failed)} failed")
    for f in failed[:10]:
        print("  failed:", *f)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "create" and len(sys.argv) > 2:
        create(sys.argv[2].split(","), sys.argv[3].split(",") if len(sys.argv) > 3 else None)
    else:
        {"setup": setup, "input": make_input, "create": create, "status": status, "fetch": fetch, "parse": parse}[cmd]()
