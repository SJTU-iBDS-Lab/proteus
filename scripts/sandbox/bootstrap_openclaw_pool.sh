#!/usr/bin/env bash
# Provision N OpenClaw sandbox containers.  Idempotent: existing
# healthy containers are left alone; missing/dead ones are (re)created
# and configured.
#
# Reads:
#   configs/sandbox/pool.yaml             — pool shape (count, image, prefix, allowlist tier)
#   configs/models.yaml                   — sandbox.target model id
#   configs/sandbox/allowlist_<TIER>.txt  — exec allowlist (one path per line)
#   .env                                  — LLM_API_KEY (passed to litellm proxy)
#
# Usage:
#   bash scripts/sandbox/bootstrap_openclaw_pool.sh
#       # uses configs/sandbox/pool.yaml + configs/models.yaml
#
#   bash scripts/sandbox/bootstrap_openclaw_pool.sh path/to/pool.yaml
#       # override pool config (e.g. different prefix for a parallel pool)
#
# Override count from cli (overrides pool.yaml):
#   COUNT=10 bash scripts/sandbox/bootstrap_openclaw_pool.sh
#
# Requires: yq, docker.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

POOL_CFG="${1:-configs/sandbox/pool.yaml}"
MODELS_CFG="${MODELS_CFG:-configs/models.yaml}"

command -v yq    >/dev/null || { echo "ERROR: yq not found (install: pip install yq, or: snap install yq)" >&2; exit 2; }
command -v docker>/dev/null || { echo "ERROR: docker not found" >&2; exit 2; }
[ -f "$POOL_CFG"   ] || { echo "ERROR: pool config not found: $POOL_CFG" >&2; exit 2; }
[ -f "$MODELS_CFG" ] || { echo "ERROR: models config not found: $MODELS_CFG" >&2; exit 2; }

# .env loading — only LLM_API_KEY is required (proxy upstream auth).
if [ -f .env ]; then
  # shellcheck disable=SC1091
  set -a; . .env; set +a
fi
[ -n "${LLM_API_KEY:-}" ] || { echo "ERROR: LLM_API_KEY must be set (in .env or env). Required by litellm proxy." >&2; exit 1; }

# ---------------------------------------------------------------------
# Read configs via yq
# ---------------------------------------------------------------------
PREFIX=$(yq    -r '.prefix'          "$POOL_CFG")
IMAGE=$(yq     -r '.image'           "$POOL_CFG")
DEFAULT_COUNT=$(yq -r '.count'       "$POOL_CFG")
LITELLM_URL=$(yq -r '.litellm.url'   "$POOL_CFG")
LITELLM_KEY=$(yq -r '.litellm.key'   "$POOL_CFG")
TIER=$(yq      -r '.allowlist_tier'  "$POOL_CFG")
TARGET_ID=$(yq -r '.sandbox.target'  "$MODELS_CFG")

COUNT="${COUNT:-$DEFAULT_COUNT}"
ALLOWLIST_FILE="$REPO_ROOT/configs/sandbox/allowlist_${TIER}.txt"
[ -f "$ALLOWLIST_FILE" ] || { echo "ERROR: allowlist tier file not found: $ALLOWLIST_FILE" >&2; exit 2; }

echo "[bootstrap] pool=${PREFIX} count=${COUNT} image=${IMAGE}" >&2
echo "[bootstrap] target_model=${TARGET_ID} via ${LITELLM_URL}" >&2
echo "[bootstrap] allowlist_tier=${TIER} (${ALLOWLIST_FILE})" >&2

# ---------------------------------------------------------------------
# Build the openclaw provider JSON once — same blob for every container.
# ---------------------------------------------------------------------
PROVIDER_JSON=$(jq -nc \
  --arg url "$LITELLM_URL" --arg key "$LITELLM_KEY" --arg id "$TARGET_ID" \
  '{baseUrl:$url, apiKey:$key, auth:"api-key", api:"openai-completions",
    models:[{id:$id, name:($id|ascii_upcase),
             api:"openai-completions", input:["text"],
             contextWindow:128000, maxTokens:12000}]}')

AUTH_JSON=$(jq -nc --arg key "$LITELLM_KEY" \
  '{openai:{default:{provider:"openai", mode:"api_key", apiKey:$key}}}')

MODELS_JSON=$(jq -nc --argjson p "$PROVIDER_JSON" \
  '{providers:{openai:$p}}')

# ---------------------------------------------------------------------
# Provision each container.  Each container is provisioned in its own
# background subshell; we wait at the end so the script blocks until
# the whole pool is ready.  Parallel docker run is safe (the daemon
# serialises layer fetches internally).
# ---------------------------------------------------------------------
provision_one() {
  local NAME="$1"

  # 1. Start container if missing or stopped
  if ! docker inspect "$NAME" >/dev/null 2>&1; then
    echo "[${NAME}] creating container..." >&2
    docker run -d --name "$NAME" "$IMAGE" sleep infinity >/dev/null
  elif [ "$(docker inspect -f '{{.State.Running}}' "$NAME")" != "true" ]; then
    echo "[${NAME}] starting existing container..." >&2
    docker start "$NAME" >/dev/null
  fi

  # 2. Write auth + models json into container
  docker exec "$NAME" mkdir -p /root/.openclaw/agents/main/agent
  printf '%s' "$AUTH_JSON"   | docker exec -i "$NAME" tee /root/.openclaw/agents/main/agent/auth-profiles.json >/dev/null
  printf '%s' "$MODELS_JSON" | docker exec -i "$NAME" tee /root/.openclaw/agents/main/agent/models.json >/dev/null

  # 3. Apply openclaw config (defaults + replace mode + provider)
  docker exec "$NAME" openclaw config set agents.defaults.model "openai/${TARGET_ID}" >/dev/null 2>&1
  docker exec "$NAME" openclaw config set models.mode replace >/dev/null 2>&1
  docker exec "$NAME" openclaw config set --strict-json models.providers.openai "$PROVIDER_JSON" >/dev/null 2>&1

  # 4. Install allowlist (one binary per line, '#' = comment)
  while IFS= read -r line || [ -n "$line" ]; do
    case "$line" in
      ''|\#*) continue ;;
      *)      docker exec "$NAME" openclaw approvals allowlist add --agent main "$line" >/dev/null 2>&1 || true ;;
    esac
  done < "$ALLOWLIST_FILE"

  # 5. Smoke probe.  ``openclaw capability model auth status`` exits 1
  #    even on success when no provider has OAuth configured, so we
  #    capture raw and re-parse rather than using a pipe (pipefail
  #    would propagate the openclaw exit and fire the `||` fallback
  #    even when jq succeeds).
  local RAW STATUS
  RAW=$(docker exec "$NAME" timeout 5 openclaw capability model auth status 2>/dev/null || true)
  if [ -n "$RAW" ]; then
    STATUS=$(printf '%s' "$RAW" | jq -rc '"providers=\([.auth.providers[]?.provider]) missing=\([.auth.missingProvidersInUse[]?]) default=\(.defaultModel // "null")"' 2>/dev/null || echo "(jq parse failed)")
  else
    STATUS="(probe returned empty)"
  fi
  echo "[${NAME}] ${STATUS}" >&2
}

export -f provision_one
export AUTH_JSON MODELS_JSON PROVIDER_JSON IMAGE TARGET_ID ALLOWLIST_FILE

# Spawn all in parallel; wait at the end.  ``set -e`` does not abort
# on individual subshell failures (would need ``wait -n`` per pid),
# but each subshell logs to stderr so failures are visible.
PIDS=()
for i in $(seq 0 $((COUNT - 1))); do
  NAME=$(printf "%s-%02d" "$PREFIX" "$i")
  provision_one "$NAME" &
  PIDS+=($!)
done

# Wait for all background jobs; collect any non-zero exit codes.
FAILED=0
for pid in "${PIDS[@]}"; do
  wait "$pid" || FAILED=$((FAILED + 1))
done
[ "$FAILED" -eq 0 ] || echo "[bootstrap] WARNING: $FAILED container(s) reported errors during provisioning" >&2

echo "[bootstrap] done. Pool ready (${COUNT} containers, ${TIER} allowlist, target=${TARGET_ID})." >&2
echo "[bootstrap] verify:  docker ps --filter name=${PREFIX} --format 'table {{.Names}}\t{{.Status}}'" >&2
