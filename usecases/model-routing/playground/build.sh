#!/usr/bin/env bash
# Build the Lambda bundle (app.py + a current boto3, for the bedrock-agentcore client) and copy the
# benchmark summary next to the web page. Run before `npx cdk deploy`.
set -euo pipefail
cd "$(dirname "$0")"
rm -rf backend/build && mkdir -p backend/build
cp backend/app.py backend/build/
python3 -m pip install -q --target backend/build "boto3>=1.43" --platform manylinux2014_aarch64 --only-binary=:all: --python-version 3.12
cp ../results/summary.json web/summary.json
echo "built backend/build and web/summary.json"
