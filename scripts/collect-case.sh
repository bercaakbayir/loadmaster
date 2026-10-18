#!/usr/bin/env bash
# Collect one ground-truth case for validation/cases.toml: start llama-server with full
# offload at the given context, record whether it loaded and what buffers it allocated.
#
#   scripts/collect-case.sh MODEL.gguf CTX [KV]      KV: f16 (default) | q8_0 | q4_0
#
# On a dedicated GPU an out-of-memory start is just a failed process. On unified memory
# (Apple Silicon) it can take the whole machine down, so there the script refuses any case
# loadmaster predicts will not fit (ADR 0005). Pass --unsafe-unified to override; don't.
set -euo pipefail

unsafe=0
args=()
for a in "$@"; do
  if [[ $a == --unsafe-unified ]]; then unsafe=1; else args+=("$a"); fi
done
[[ ${#args[@]} -ge 2 ]] || { sed -n '2,10p' "$0"; exit 2; }
model=${args[0]} ctx=${args[1]} kv=${args[2]:-f16}
port=${PORT:-18744}
lm=${LOADMASTER:-loadmaster}

plan=$("$lm" plan "$model" --ctx "$ctx" --json || true)
predicted=$(python3 -c 'import json,sys; print("fits" if json.loads(sys.stdin.read())["result"]=="fits" else "oom")' <<<"$plan")
unified=$("$lm" gpu --json | python3 -c 'import json,sys; g=json.load(sys.stdin)["gpus"]; print(int(bool(g) and g[0]["memory"]=="unified"))')
if [[ $unified == 1 && $predicted != fits && $unsafe == 0 ]]; then
  echo "refusing: unified memory and loadmaster predicts this does not fit; an OOM here can reset the machine" >&2
  exit 3
fi

log=$(mktemp)
trap 'rm -f "$log"' EXIT
llama-server -m "$model" -c "$ctx" -ngl 999 -fa on -ctk "$kv" -ctv "$kv" -np 1 -lv 4 --port "$port" >"$log" 2>&1 &
pid=$!
truth=oom
for _ in $(seq 1 600); do
  if grep -q "listening on" "$log"; then truth=fits; break; fi
  if ! kill -0 "$pid" 2>/dev/null; then break; fi
  if grep -qE "failed to allocate|out of memory|failed to load model|exiting due to" "$log"; then break; fi
  sleep 0.5
done
kill "$pid" 2>/dev/null || true
wait "$pid" 2>/dev/null || true

sum() { grep -E "$1" "$log" | sed -E 's/.*= *([0-9.]+) MiB.*/\1/' | awk '{s+=$1} END {printf "%.2f", s}'; }
gpu_name=$("$lm" gpu --json | python3 -c 'import json,sys; g=json.load(sys.stdin)["gpus"]; print(g[0]["name"] if g else "none")')
version=$(llama-server --version 2>&1 | grep -m1 version || true)
cat <<TOML
[[case]]
id = "$(basename "$model" .gguf | tr 'A-Z' 'a-z')-$(echo "$gpu_name" | tr 'A-Z ' 'a-z-')-${ctx}-${kv}"
model = "$(basename "$model")"
sha256 = "$(shasum -a 256 "$model" | cut -d' ' -f1)"
runtime = "llama.cpp"
ctx = $ctx
kv = "$kv"
hw = $("$lm" gpu --json | python3 -c '
import json,sys
s=json.load(sys.stdin); g=s["gpus"][0]; M=1<<20
print("{ name = %s, vendor = \"%s\", memory = \"%s\", total_mib = %d, usable_mib = %d, host_available_mib = %d }" % (json.dumps(g["name"]), g["vendor"], g["memory"], g["total_bytes"]//M, g["usable_bytes"]//M, s["host"]["available_bytes"]//M))')
truth = "$truth"
# loadmaster predicted: $predicted; $version
# llama.cpp reported (MiB): gpu model $(sum 'MTL0_Mapped model|CUDA0 model|ROCm0 model'), kv $(sum '(MTL0|CUDA0|ROCm0) KV buffer'), compute $(sum '(MTL0|CUDA0|ROCm0) compute buffer')
TOML
