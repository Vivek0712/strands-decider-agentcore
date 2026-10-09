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

APO_MODELS = ["micro", "lite", "pro", "scout", "llama8b"]
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
        lines.append(
            {
                "version": "bedrock-2026-05-14",
                "templateId": fam,
                "promptTemplate": tpl,
                "customEvaluationMetricLabel": f"{fam}exactscore",
                "evaluationMetricLambdaArn": arn,
                "evaluationSamples": samples,
            }
        )
    path = ROOT / "data" / "apo_input.jsonl"
    path.write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    session.client("s3").upload_file(str(path), BUCKET, "input/apo_input.jsonl")
    print(f"{len(lines)} templates, {sum(len(x['evaluationSamples']) for x in lines)} samples -> s3://{BUCKET}/input/")


def create() -> None:
    br = session.client("bedrock")
    name = f"decider-routing-{time.strftime('%Y%m%d-%H%M%S')}"
    arn = br.create_advanced_prompt_optimization_job(
        jobName=name,
        jobDescription="Model selection for 6 task families (decider routing benchmark)",
        modelConfigurations=[
            {"modelId": MODELS[m].model_id, "inferenceConfig": {"temperature": 0.0, "maxTokens": 1500}}
            for m in APO_MODELS
        ],
        inputConfig={"s3Uri": f"s3://{BUCKET}/input/apo_input.jsonl"},
        outputConfig={"s3Uri": f"s3://{BUCKET}/output/"},
    )["jobArn"]
    STATE.write_text(json.dumps({"jobArn": arn, "jobName": name, "created": time.time()}, indent=2))
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
    job = json.loads(STATE.read_text())["jobArn"].split("/")[-1]
    path = next((ROOT / "results" / "apo_raw").glob(f"*{job}*results.jsonl"))
    prompts: dict = {}
    scores: dict = {}
    failed = []
    for line in path.read_text().splitlines():
        rec = json.loads(line)
        fam = rec["promptTemplateId"]
        for r in rec["promptOptimizationResults"]:
            m = by_id.get(r["modelId"], r["modelId"])
            if r["status"] != "COMPLETED" and not r.get("optimizedPromptMetrics", {}).get("averageScore"):
                failed.append((fam, m, r.get("failureReason", "")[:120]))
                continue
            prompts.setdefault(fam, {})[m] = r["optimizedPromptTemplate"]
            scores.setdefault(fam, {})[m] = r["optimizedPromptMetrics"]["averageScore"]
    (ROOT / "results" / "apo_prompts.json").write_text(json.dumps(prompts, indent=2))
    (ROOT / "results" / "apo_scores.json").write_text(json.dumps(scores, indent=2))
    print(f"{sum(len(v) for v in scores.values())} optimized (family, model) pairs; {len(failed)} failed")
    for f in failed[:10]:
        print("  failed:", *f)


if __name__ == "__main__":
    {"setup": setup, "input": make_input, "create": create, "status": status, "fetch": fetch, "parse": parse}[
        sys.argv[1]
    ]()
