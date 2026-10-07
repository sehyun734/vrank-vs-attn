#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

session="$(openssl rand -hex 3)"
tar_path="/tmp/$session.tar.gz"
argv="$(python3 -c "import sys; print(sys.argv[1:])" "$@")"

entry_name="$(basename "$1" .py)"
model_name_pattern=" --model_name[= ]([^ ]+)"
if [[ " $* " =~ $model_name_pattern ]]; then
  model_name="${BASH_REMATCH[1]}"
else
  model_name="$(sed -nE 's/^ *model_name: str = "([^"]+)".*/\1/p' "$1")"
fi
log_path="logs/$entry_name/${model_name##*/}/$(date +%y%m%d-%H%M%S).log"
mkdir -p "$(dirname "$log_path")"

trap 'rm -f "$tar_path"; colab stop -s "$session" || true' EXIT
colab new -s "$session" --gpu A100
COPYFILE_DISABLE=1 tar -czf "$tar_path" --exclude .venv --exclude __pycache__ --exclude logs .
colab upload -s "$session" "$tar_path" content/project.tar.gz
colab install -s "$session" $(grep -E "^(accelerate|datasets|huggingface_hub|safetensors|simple-parsing|transformers)==" requirements.txt)

colab exec -s "$session" --timeout 86400 <<EOF 2>&1 | tee "$log_path"
import subprocess
import tarfile
tarfile.open("project.tar.gz").extractall(filter="data")
proc = subprocess.Popen(
    ["bash", "-c", 'set -a && . ./.env && set +a && PYTHONPATH=. exec python -u "\$@"', "bash", *$argv],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
)
for line in proc.stdout:
    print(line, end="", flush=True)
if proc.wait() != 0:
    print("fail", flush=True)
EOF
