from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize

from core.models import SkillFingerprintRecord
from shared.schemas.subgraph_schema import skill_fingerprint_from_record

from .subgraph_io import write_jsonl_records

ALL_API_KEYS: list[str] = [
    "os.getenv", "os.environ.get", "input",
    "sys.stdin.read", "sys.stdin.readline",
    "request.json", "request.form", "request.get_json",
    "subprocess.run", "subprocess.call", "subprocess.Popen",
    "subprocess.check_output", "subprocess.check_call",
    "os.system", "os.popen",
    "eval", "exec", "compile",
    "__import__", "importlib.import_module",
    "cursor.execute", "cursor.executemany",
    "connection.execute", "session.execute",
    "open_read", "open_write",
    "requests.get", "requests.post", "requests.put",
    "httpx.get", "httpx.post",
    "json.dumps", "json.loads", "json.load", "json.dump",
    "base64.b64encode", "base64.b64decode",
    "pickle.dumps", "pickle.loads",
    "yaml.safe_load", "yaml.dump",
]


def _api_features_to_vector(api_features: dict[str, int]) -> list[float]:
    return [1.0 if api_features.get(k, 0) > 0 else 0.0 for k in ALL_API_KEYS]


def build_feature_matrix(
    records: list[SkillFingerprintRecord],
) -> np.ndarray:
    rows: list[np.ndarray] = []
    for rec in records:
        embedding = np.array(rec.doc_embedding, dtype=np.float32)
        api_vec = np.array(_api_features_to_vector(rec.api_features), dtype=np.float32)
        # Weight api_vec to match embedding magnitude (~1.0 per dim after normalize)
        api_vec *= 3.0
        rows.append(np.concatenate([embedding, api_vec]))
    matrix = np.stack(rows)
    return normalize(matrix)


def cluster_fingerprints(
    records: list[SkillFingerprintRecord],
    *,
    n_clusters: int = 500,
    seed: int = 42,
) -> list[int]:
    matrix = build_feature_matrix(records)
    kmeans = MiniBatchKMeans(
        n_clusters=min(n_clusters, len(records)),
        random_state=seed,
        batch_size=1024,
        n_init=3,
    )
    labels = kmeans.fit_predict(matrix)
    return [int(x) for x in labels]


def _risk_score(rec: SkillFingerprintRecord) -> int:
    return len(rec.source_tags) + len(rec.sink_tags) * 2


def select_candidates(
    records: list[SkillFingerprintRecord],
    labels: list[int],
    *,
    max_per_cluster: int = 15,
) -> list[SkillFingerprintRecord]:
    clusters: dict[int, list[tuple[int, SkillFingerprintRecord]]] = {}
    for idx, (rec, label) in enumerate(zip(records, labels)):
        clusters.setdefault(label, []).append((idx, rec))

    selected: list[SkillFingerprintRecord] = []
    for _label, members in sorted(clusters.items()):
        if len(members) <= max_per_cluster:
            for _idx, rec in members:
                selected.append(rec)
            continue
        ranked = sorted(members, key=lambda x: _risk_score(x[1]), reverse=True)
        for _idx, rec in ranked[:max_per_cluster]:
            selected.append(rec)

    return selected


def run_cluster_pipeline(
    fingerprints_path: Path,
    output_dir: Path,
    *,
    n_clusters: int = 500,
    max_per_cluster: int = 15,
) -> dict[str, Path]:
    records: list[SkillFingerprintRecord] = []
    with fingerprints_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            records.append(skill_fingerprint_from_record(json.loads(line)))

    print(f"loaded {len(records)} fingerprints")

    labels = cluster_fingerprints(
        records, n_clusters=n_clusters,
    )
    print(f"clustered into {len(set(labels))} clusters")

    for rec, label in zip(records, labels):
        rec.cluster_id = f"cluster-{label:04d}"

    cluster_rows = [
        {"skill_id": rec.skill_id, "cluster_id": rec.cluster_id}
        for rec in records
    ]
    cluster_path = write_jsonl_records(
        output_dir / "cluster_assignments.jsonl", cluster_rows,
    )

    candidates = select_candidates(records, labels, max_per_cluster=max_per_cluster)
    print(f"selected {len(candidates)} candidates from {len(set(labels))} clusters")

    candidate_path = write_jsonl_records(
        output_dir / "candidate_skill_set.jsonl",
        [rec.to_dict() for rec in candidates],
    )

    cluster_sizes = Counter(labels)
    stats = {
        "total_fingerprints": len(records),
        "n_clusters": len(set(labels)),
        "total_candidates": len(candidates),
        "max_per_cluster": max_per_cluster,
        "cluster_size_min": min(cluster_sizes.values()),
        "cluster_size_max": max(cluster_sizes.values()),
        "cluster_size_median": sorted(cluster_sizes.values())[len(cluster_sizes) // 2],
    }
    stats_path = output_dir / "cluster_stats.json"
    stats_path.write_text(
        json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )

    print(f"cluster stats: {stats}")
    return {
        "cluster_assignments": cluster_path,
        "candidate_skill_set": candidate_path,
        "cluster_stats": stats_path,
    }
