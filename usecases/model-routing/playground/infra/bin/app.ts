#!/usr/bin/env node
import * as fs from "fs";
import * as path from "path";
import * as crypto from "crypto";
import { App } from "aws-cdk-lib";
import { PlaygroundStack } from "../lib/stack";

const app = new App();
const deciderArn = app.node.tryGetContext("deciderArn") ?? process.env.DECIDER_ARN;
if (!deciderArn) throw new Error("pass -c deciderArn=<the decider runtime ARN> (python deciderctl.py list)");

// a per-checkout secret that CloudFront adds to every /api request; the Lambda rejects requests without it
const secretFile = path.join(__dirname, "..", ".origin-secret");
if (!fs.existsSync(secretFile)) fs.writeFileSync(secretFile, crypto.randomBytes(24).toString("hex"));

new PlaygroundStack(app, "DeciderModelRoutingPlayground", {
  env: { account: process.env.CDK_DEFAULT_ACCOUNT, region: process.env.CDK_DEFAULT_REGION ?? "us-east-1" },
  deciderArn,
  originSecret: fs.readFileSync(secretFile, "utf8").trim(),
  perIpHour: Number(app.node.tryGetContext("perIpHour") ?? 40),
  globalDay: Number(app.node.tryGetContext("globalDay") ?? 1500),
  keepWarm: (app.node.tryGetContext("keepWarm") ?? "true") === "true",
});
