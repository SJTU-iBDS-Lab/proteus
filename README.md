# PROTEUS

Official code release for **PROTEUS**, accepted at **NeurIPS 2026**.

PROTEUS is a graph-guided red-team evolution framework for evaluating the
security of LLM agents that consume third-party skills. It combines structured
skill mutation, graph-guided expansion, sandbox execution, lethality oracles,
and blue-team auditing to discover attacks that are both **lethal** and
**evasive**.

## Core execution model

With the lethal-gated audit configuration, each campaign round proceeds as
follows:

1. **Mutate** the current skill variant.
2. **Materialize** the exact candidate artifact, including any sandbox-required
   rewrite.
3. **Execute** that candidate in the OpenClaw sandbox and evaluate its
   lethality oracle.
4. **Audit only lethal candidates.** Non-lethal candidates do not invoke the
   auditor and are recorded as `audit_verdict=NOT_RUN`.
5. Record joint success only when the **same candidate** is both
   `bypassed=True` and `lethal=True`.
6. Optionally continue with graph-guided path and surface expansion.

This ordering avoids spending auditor tokens on candidates that cannot produce
a joint success and prevents parent/child outcome mismatches.

## Repository layout

| Path | Contents |
| --- | --- |
| `src/red_team/agent/` | Tool-using mutation agent and runtime |
| `src/red_team/campaign/` | Campaign loop, run state, audit gating, and execution seams |
| `src/red_team/evolution/` | Evolution driver and horizontal expansion |
| `src/red_team/mutation/` | Mutation strategies and search-space constraints |
| `src/red_team/baselines/` | Programmatic baselines and PyRIT integration |
| `src/red_team/external_baselines/` | SKILL-INJECT Bo5 and SkillJect adapters |
| `src/sandbox/` | OpenClaw execution, poisoned-chain materialization, and sensors |
| `src/blue_team/` | Blue-team auditing components |
| `src/gray_box/` | AI-Infra-Guard scanner integration |
| `scripts/red_team/` | Single-seed and external-baseline entry points |
| `scripts/sandbox/` | Sandbox image and container-pool provisioning |
| `tests/` | Unit and integration tests |

## Entry points

### Single campaign

`run_evolution.py` executes one seed/repository pair:

```bash
export LITELLM_URL="http://127.0.0.1:4000"
export LITELLM_KEY="your-local-proxy-key"

PYTHONPATH=src uv run --no-sync python scripts/red_team/run_evolution.py \
  --seed SEED_ID \
  --repo REPO_ID \
  --seeds-file /path/to/seeds.jsonl \
  --targets-file /path/to/targets.jsonl \
  --target-model TARGET_MODEL \
  --mutator-model MUTATOR_MODEL \
  --blue-team aig \
  --sandbox-container CONTAINER_NAME \
  --ablation-json '{"audit_gate":"lethal"}' \
  --rounds 5 \
  --out outputs/proteus
```

Use `--help` for the full set of campaign and ablation options.

### External baselines

The unified baseline runner supports PyRIT standalone, SKILL-INJECT Bo5, and
SkillJect:

```bash
SEEDS=/path/to/seeds.jsonl TOTAL=20 ROUNDS=5 \
  bash scripts/red_team/run_external_baselines.sh \
  outputs/external_baselines all
```

Set `STRATEGY` through the second positional argument to run one method:
`pyrit_standalone`, `skill_inject_bo5`, or `skillject_trace`. `START` and `END`
select an inclusive, 1-based seed range. The PyRIT arm requires `pyrit==0.12.1`
in the selected Python environment.

## Outputs

Campaign runs write structured artifacts including:

- `log.jsonl` — per-round outcomes and execution metadata;
- `summary.json` — aggregate campaign statistics;
- `harvest.jsonl` — successful-variant harvest records;
- `attack_log.jsonl` — cross-round attack memory when enabled.

The batch baseline runner additionally writes `batch_summary.json` for each
strategy.

## Requirements

- Python 3.13 and `uv`;
- Docker and an OpenClaw runtime for sandbox execution;
- a LiteLLM-compatible proxy exposing the model IDs in `configs/models.yaml`;
- AI-Infra-Guard when using `--blue-team aig`;
- PyRIT 0.12.1 only for the PyRIT baseline.

Experiment datasets, downloaded skills, provider credentials, and local
provider-routing files are not distributed with this repository.

## Tests

```bash
uv run --no-sync pytest -q
```

The test suite covers campaign semantics, lethal-gated auditing, mutation and
expansion constraints, sandbox/oracle interfaces, baseline adapters, and
structured output handling.

## Citation

If you use PROTEUS in your research, please cite the **NeurIPS 2026** paper.
