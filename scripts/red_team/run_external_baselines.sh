#!/usr/bin/env bash
# Unified external-baseline runner: PyRIT, SKILL-INJECT Bo5-Line, SkillJect.
# All methods share seeds, candidate budget, target/adversary models, sandbox,
# AIG audit gate, and structured-log aggregation.
#
# Usage:
#   TARGET_MODEL=glm-5 ATTACK_MODEL=deepseek-v4-pro \
#     bash scripts/red_team/run_external_baselines.sh [out_root] [strategy]
#
# strategy: all (default) | both (Bo5 + SkillJect; legacy behavior) |
#           pyrit_standalone | skill_inject_bo5 | skillject_trace
# Model, seed, pool, and budget defaults can be overridden via environment.
# Use START/END for an inclusive 1-based seed range (for example START=11 END=20).
# The PyRIT arm requires pyrit==0.12.1 in the selected PYTHON environment.
set -euo pipefail
cd "$(dirname "$0")/../.."

TS="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${1:-outputs/external_baselines_${TS}}"
STRATEGY="${2:-all}"
mkdir -p "$OUT_ROOT"

PYTHON="${PYTHON:-.venv/bin/python}"
SEEDS="${SEEDS:-data/seeds/malicious_seeds.jsonl}"
CONCURRENCY="${CONCURRENCY:-8}"
TOTAL="${TOTAL:-20}"
START="${START:-1}"
END="${END:-$TOTAL}"
ROUNDS="${ROUNDS:-5}"
TARGET_MODEL="${TARGET_MODEL:-deepseek-v4-flash}"
ATTACK_MODEL="${ATTACK_MODEL:-deepseek-v4-pro}"
BLUE_TEAM="${BLUE_TEAM:-aig}"
AUDIT_GATE="${AUDIT_GATE:-lethal}"
CPREFIX="${CPREFIX:-skillrt-openclaw}"

if ((START < 1 || END < START || CONCURRENCY < 1 || ROUNDS < 1)); then
  echo "invalid range/budget: START=$START END=$END CONCURRENCY=$CONCURRENCY ROUNDS=$ROUNDS" >&2
  exit 2
fi

export PYRIT_TARGET_MODEL="$TARGET_MODEL"
export PYRIT_ADVERSARY_MODEL="$ATTACK_MODEL"
export SKILLJECT_MODEL="$ATTACK_MODEL"
export LITELLM_URL="${LITELLM_URL:-http://127.0.0.1:4001}"
export LITELLM_KEY="${LITELLM_KEY:-sk-claw-local}"
export AIG_PYTHON="${AIG_PYTHON:-$PYTHON}"

echo "=== external baselines: ${ATTACK_MODEL} -> ${TARGET_MODEL} | ${BLUE_TEAM} | gate=${AUDIT_GATE} ==="
echo "seeds=${SEEDS} range=${START}..${END} rounds=${ROUNDS} concurrency=${CONCURRENCY}"

mapfile -t seed_lines < <(
  "$PYTHON" - "$SEEDS" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if path.suffix.lower() == ".json":
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("seeds") if isinstance(payload, dict) else payload
else:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
for row in rows:
    sid = row.get("seed_id") or row.get("id")
    chain = row.get("chain") or []
    print(f"{sid}|{chain[0] if chain else ''}")
PY
)

launch_seed() {
  local strategy="$1" idx="$2"
  local idx0=$((idx - 1))
  local line="${seed_lines[$idx0]:-}"
  [ -z "$line" ] && { echo "  seed $idx not found"; return; }
  local sid="${line%%|*}"
  local repo="${line#*|}"
  local cidx
  cidx=$(printf "%02d" $((idx0 % CONCURRENCY)))
  local container="${CPREFIX}-${cidx}"
  local outdir="${OUT_ROOT}/${strategy}/${sid}_${repo}"
  local console="${OUT_ROOT}/${strategy}/${sid}_${repo}.console"
  mkdir -p "$outdir"
  echo "  [$(date '+%H:%M:%S')] launch $strategy $sid -> $container"
  (
    LLM_REQUEST_TIMEOUT=300 PYTHONPATH=src \
      "$PYTHON" scripts/red_team/run_evolution.py \
        --seed "$sid" --repo "$repo" --seeds-file "$SEEDS" \
        --target-model "$TARGET_MODEL" --mutator-model "$ATTACK_MODEL" \
        --rounds "$ROUNDS" --blue-team "$BLUE_TEAM" \
        --sandbox-container "$container" \
        --ablation-json "{\"red_strategy\":\"${strategy}\",\"audit_gate\":\"${AUDIT_GATE}\",\"phase\":\"main_only\",\"disable_expansion\":true}" \
        --out "$outdir"
  ) > "$console" 2>&1 &
  PIDS+=($!)
  sleep 1
}

run_strategy() {
  local strategy="$1"
  if [ "$strategy" = "pyrit_standalone" ]; then
    "$PYTHON" -I - <<'PY'
from importlib.metadata import PackageNotFoundError, version

expected = "0.12.1"
try:
    installed = version("pyrit")
except PackageNotFoundError:
    raise SystemExit("PyRIT baseline requires pyrit==0.12.1 in PYTHON")
if installed != expected:
    raise SystemExit(f"PyRIT baseline requires {expected}, found {installed}")
PY
  fi
  echo
  echo "=== strategy ${strategy}: seeds ${START}..${END}, ${CONCURRENCY} concurrent ==="
  mkdir -p "${OUT_ROOT}/${strategy}"
  local count=$((END - START + 1))
  local waves=$(( (count + CONCURRENCY - 1) / CONCURRENCY ))
  for ((w=1; w<=waves; w++)); do
    local start=$(( START + (w - 1) * CONCURRENCY ))
    local end=$((start + CONCURRENCY - 1))
    [ "$end" -gt "$END" ] && end="$END"
    local log="${OUT_ROOT}/${strategy}/wave${w}.log"
    echo "=== wave ${w}/${waves}: seeds ${start}..${end} ==="
    PIDS=()
    {
      for ((i=start; i<=end; i++)); do
        launch_seed "$strategy" "$i"
      done
      echo "[wave ${w}] launched ${#PIDS[@]} jobs; waiting..."
      fail=0
      for pid in "${PIDS[@]}"; do
        wait "$pid" || fail=$((fail + 1))
      done
      echo "[wave ${w}] DONE  ok=$(( ${#PIDS[@]} - fail ))  fail=${fail}"
    } > "$log" 2>&1
    tail -1 "$log"
    echo "    wave ${w} finished $(date '+%H:%M:%S')"
  done
}

summarize() {
  local strategy="$1"
  echo "--- ${strategy} ---"
  "$PYTHON" - "$OUT_ROOT/$strategy" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
runs = []
for log_path in sorted(out.glob("seed-*/seed-*/log.jsonl")):
    rounds = bypass = lethal = star = 0
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("is_expansion"):
            continue
        rounds += 1
        b = row.get("bypassed") is True
        l = row.get("lethal") is True
        bypass += b
        lethal += l
        star += b and l
    runs.append({
        "run_dir": str(log_path.parent.relative_to(out)),
        "rounds": rounds,
        "bypass": bypass,
        "lethal": lethal,
        "star": star,
    })

summary = {
    "strategy": out.name,
    "seeds_total": len(runs),
    "seeds_with_star": sum(run["star"] > 0 for run in runs),
    "total_rounds": sum(run["rounds"] for run in runs),
    "total_bypass": sum(run["bypass"] for run in runs),
    "total_lethal": sum(run["lethal"] for run in runs),
    "total_star": sum(run["star"] for run in runs),
    "runs": runs,
}
(out / "batch_summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
for run in runs:
    print(
        f"  {run['run_dir'][:55]:55s} rounds={run['rounds']:2d} "
        f"b={run['bypass']:2d} l={run['lethal']:2d} star={run['star']:2d}"
    )
print(
    f"  TOTAL rounds={summary['total_rounds']} b={summary['total_bypass']} "
    f"l={summary['total_lethal']} star={summary['total_star']} "
    f"seeds_with_star={summary['seeds_with_star']}/{summary['seeds_total']}"
)
PY
}

run_and_summarize() {
  run_strategy "$1"
  summarize "$1"
}

case "$STRATEGY" in
  all)
    run_and_summarize pyrit_standalone
    run_and_summarize skill_inject_bo5
    run_and_summarize skillject_trace
    ;;
  both)
    run_and_summarize skill_inject_bo5
    run_and_summarize skillject_trace
    ;;
  pyrit_standalone|skill_inject_bo5|skillject_trace)
    run_and_summarize "$STRATEGY"
    ;;
  *) echo "unknown strategy: $STRATEGY" >&2; exit 2 ;;
esac

echo
echo "=== ALL DONE  $(date '+%H:%M:%S')  out: ${OUT_ROOT} ==="
