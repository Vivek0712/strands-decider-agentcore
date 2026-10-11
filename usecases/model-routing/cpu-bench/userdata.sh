#!/bin/bash
# Boot script for the CPU benchmark instance: install, serve, benchmark bf16 then int8, upload, terminate.
exec > /var/log/bench.log 2>&1
set -x
shutdown -h +100   # hard safety limit: the instance terminates itself after 100 minutes no matter what
BUCKET=__BUCKET__; PREFIX=__PREFIX__; QENGINE=__QENGINE__
export DEBIAN_FRONTEND=noninteractive
apt-get update -y && apt-get install -y python3-venv python3-dev git build-essential
python3 -m venv /opt/v && . /opt/v/bin/activate
pip install -q --upgrade pip
pip install -q torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
pip install -q "bedrock-agentcore>=1.18.1" "huggingface_hub>=0.24" boto3 \
  "strands-decider @ git+https://github.com/strands-labs/strands-decider@3e94e9d84c620ed5a95f1a3310c3decb971e261c"
git clone -q https://github.com/Vivek0712/strands-decider-agentcore /opt/sda
aws_up() { python -c "import boto3,sys; boto3.client('s3').upload_file(sys.argv[1], '$BUCKET', '$PREFIX/'+sys.argv[2])" "$1" "$2"; }
cd /opt/sda/runtime
export DECIDER_MODEL=StrandsAgents/strands-decider-2B-hobson-v19 DECIDER_REVISION=bb282d786bc251fd4e3068de3ada9ddbb38127cd \
       HF_HOME=/opt/hf MALLOC_ARENA_MAX=2 MALLOC_MMAP_THRESHOLD_=1048576 MALLOC_TRIM_THRESHOLD_=1048576 TOKENIZERS_PARALLELISM=false
for QUANT in bf16 int8; do
  DECIDER_QUANT=$QUANT DECIDER_QENGINE=$QENGINE python main.py > /var/log/server-$QUANT.log 2>&1 &
  PID=$!
  for i in $(seq 1 120); do
    sleep 10
    curl -s -X POST localhost:8080/invocations -H 'Content-Type: application/json' -d '{"action":"health"}' | grep -q '"ready": true' && break
  done
  python /opt/sda/usecases/model-routing/cpu-bench/bench.py --quant $QUANT --out /opt/res-$QUANT.json
  kill $PID; sleep 8
  aws_up /opt/res-$QUANT.json res-$QUANT.json
  aws_up /var/log/server-$QUANT.log server-$QUANT.log
done
aws_up /var/log/bench.log bench.log
shutdown -h now
