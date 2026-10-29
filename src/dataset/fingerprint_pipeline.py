from __future__ import annotations

import ast
import importlib
from dataclasses import dataclass, field
from typing import Protocol

from core.models import SkillArtifact, SkillFingerprintRecord

# ── Declared-capability → tag mapping (fallback for non-Python skills) ───

_DECLARED_SOURCE_MAP: dict[str, str] = {
    "credential_access": "credentials",
    "file_read": "local_files",
}

_DECLARED_SINK_MAP: dict[str, str] = {
    "file_write": "filesystem_write",
    "shell_exec": "shell_execution",
}

# network_access is ambiguous: skill can receive OR send
_DECLARED_DUAL_MAP: dict[str, tuple[str, str]] = {
    "network_access": ("network_input", "network_egress"),
}


class DocumentEmbedder(Protocol):
    model_name: str

    def encode(self, text: str) -> list[float]: ...


class _SentenceEncoder(Protocol):
    def encode(self, text: str) -> list[float] | tuple[float, ...]: ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model: _SentenceEncoder | None = None

    def _load_model(self) -> _SentenceEncoder:
        if self._model is not None:
            return self._model
        try:
            module = importlib.import_module("sentence_transformers")
            sentence_transformer_cls = getattr(module, "SentenceTransformer")
        except (ImportError, AttributeError) as exc:  # pragma: no cover - optional dep
            raise RuntimeError(
                "sentence-transformers is required for real document embeddings. "
                "Install it with `uv sync --group dataset`."
            ) from exc
        self._model = sentence_transformer_cls(self.model_name)
        model = self._model
        assert model is not None
        return model

    def encode(self, text: str) -> list[float]:
        model = self._load_model()
        vector = model.encode(text)
        return [float(item) for item in vector]


@dataclass(slots=True)
class AstExtractionResult:
    code_ops: list[str] = field(default_factory=list)
    source_tags: list[str] = field(default_factory=list)
    sink_tags: list[str] = field(default_factory=list)
    bridge_tags: list[str] = field(default_factory=list)
    dependency_profile: list[str] = field(default_factory=list)
    api_features: dict[str, int] = field(default_factory=dict)


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        left = _call_name(node.value)
        return f"{left}.{node.attr}" if left else node.attr
    return ""


def _string_constant(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


_SOURCE_APIS: dict[str, str] = {
    "os.getenv": "env_read",
    "os.environ.get": "env_read",
    "input": "user_input",
    "sys.stdin.read": "stdin_read",
    "sys.stdin.readline": "stdin_read",
    "request.json": "http_input",
    "request.form": "http_input",
    "request.get_json": "http_input",
}

_SINK_APIS: dict[str, str] = {
    "subprocess.run": "shell_exec",
    "subprocess.call": "shell_exec",
    "subprocess.Popen": "shell_exec",
    "subprocess.check_output": "shell_exec",
    "subprocess.check_call": "shell_exec",
    "os.system": "shell_exec",
    "os.popen": "shell_exec",
    "eval": "code_exec",
    "exec": "code_exec",
    "compile": "code_exec",
    "__import__": "dynamic_import",
    "importlib.import_module": "dynamic_import",
    "cursor.execute": "sql_exec",
    "cursor.executemany": "sql_exec",
    "connection.execute": "sql_exec",
    "session.execute": "sql_exec",
}

# Network APIs split by direction:
# GET/read = pulls external data in (source)
# POST/send = pushes data out (sink)
# Ambiguous calls get both roles.
_NETWORK_RECV_NAMES: set[str] = {
    "requests.get", "requests.head", "requests.options",
    "httpx.get", "httpx.head", "httpx.options",
    "urllib.request.urlopen", "urllib.request.urlretrieve",
    "aiohttp.request",
}
_NETWORK_SEND_NAMES: set[str] = {
    "requests.post", "requests.put", "requests.patch", "requests.delete",
    "httpx.post", "httpx.put", "httpx.patch", "httpx.delete",
    "fetch",
}
# Generic session/client calls are ambiguous — mark both
_NETWORK_AMBIGUOUS_NAMES: set[str] = {
    "requests.request", "httpx.request",
}

_BRIDGE_APIS: dict[str, str] = {
    "json.dumps": "json_transform",
    "json.loads": "json_transform",
    "json.load": "json_transform",
    "json.dump": "json_transform",
    "base64.b64encode": "base64_transform",
    "base64.b64decode": "base64_transform",
    "pickle.dumps": "pickle_transform",
    "pickle.loads": "pickle_transform",
    "yaml.safe_load": "yaml_transform",
    "yaml.dump": "yaml_transform",
}

_SOURCE_TAG_RULES: dict[str, str] = {
    "env_read": "credentials",
    "user_input": "user_input",
    "stdin_read": "user_input",
    "http_input": "user_input",
    "dynamic_import": "external_code",
    "network_recv": "network_input",
}

_SINK_TAG_RULES: dict[str, str] = {
    "shell_exec": "shell_execution",
    "code_exec": "code_execution",
    "sql_exec": "sql_execution",
    "network_send": "network_egress",
}

_BRIDGE_TAG_RULES: dict[str, str] = {
    "json_transform": "json_transform",
    "base64_transform": "base64_transform",
    "pickle_transform": "pickle_transform",
    "yaml_transform": "yaml_transform",
}


def extract_python_ast_features(code: str) -> AstExtractionResult:
    tree = ast.parse(code)
    imports: set[str] = set()
    api_features: dict[str, int] = {}
    code_ops: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
        elif isinstance(node, ast.Call):
            name = _call_name(node.func)

            # Source APIs
            if name in _SOURCE_APIS:
                api_features[name] = api_features.get(name, 0) + 1
                code_ops.add(_SOURCE_APIS[name])

            # open() — special handling for read vs write mode
            if name == "open":
                mode = "r"
                if len(node.args) >= 2:
                    maybe_mode = _string_constant(node.args[1])
                    if maybe_mode is not None:
                        mode = maybe_mode
                for keyword in node.keywords:
                    if keyword.arg == "mode":
                        maybe_mode = _string_constant(keyword.value)
                        if maybe_mode is not None:
                            mode = maybe_mode
                if any(flag in mode for flag in ("w", "a", "x", "+")):
                    api_features["open_write"] = api_features.get("open_write", 0) + 1
                    code_ops.add("filesystem_write")
                else:
                    api_features["open_read"] = api_features.get("open_read", 0) + 1
                    code_ops.add("filesystem_read")

            # Sink APIs
            if name in _SINK_APIS:
                api_features[name] = api_features.get(name, 0) + 1
                code_ops.add(_SINK_APIS[name])

            # Network requests — split by direction
            if name in _NETWORK_RECV_NAMES:
                api_features[name] = api_features.get(name, 0) + 1
                code_ops.add("network_recv")
            if name in _NETWORK_SEND_NAMES:
                api_features[name] = api_features.get(name, 0) + 1
                code_ops.add("network_send")
            if name in _NETWORK_AMBIGUOUS_NAMES:
                api_features[name] = api_features.get(name, 0) + 1
                code_ops.add("network_recv")
                code_ops.add("network_send")

            # Bridge APIs
            if name in _BRIDGE_APIS:
                api_features[name] = api_features.get(name, 0) + 1
                code_ops.add(_BRIDGE_APIS[name])

    # Derive tags from code_ops
    source_tags: set[str] = set()
    sink_tags: set[str] = set()
    bridge_tags: set[str] = set()

    if "filesystem_read" in code_ops:
        source_tags.add("local_files")
    for op, tag in _SOURCE_TAG_RULES.items():
        if op in code_ops:
            source_tags.add(tag)

    if "filesystem_write" in code_ops:
        sink_tags.add("filesystem_write")
    for op, tag in _SINK_TAG_RULES.items():
        if op in code_ops:
            sink_tags.add(tag)

    for op, tag in _BRIDGE_TAG_RULES.items():
        if op in code_ops:
            bridge_tags.add(tag)

    return AstExtractionResult(
        code_ops=sorted(code_ops),
        source_tags=sorted(source_tags),
        sink_tags=sorted(sink_tags),
        bridge_tags=sorted(bridge_tags),
        dependency_profile=sorted(imports),
        api_features=api_features,
    )


def extract_skill_ast_features(artifact: SkillArtifact) -> AstExtractionResult:
    python_files = [
        content
        for path, content in artifact.code_files.items()
        if path.lower().endswith(".py")
    ]
    if not python_files:
        return AstExtractionResult()

    code_ops: set[str] = set()
    source_tags: set[str] = set()
    sink_tags: set[str] = set()
    bridge_tags: set[str] = set()
    dependency_profile: set[str] = set()
    api_features: dict[str, int] = {}

    for content in python_files:
        try:
            result = extract_python_ast_features(content)
        except SyntaxError:
            continue
        code_ops.update(result.code_ops)
        source_tags.update(result.source_tags)
        sink_tags.update(result.sink_tags)
        bridge_tags.update(result.bridge_tags)
        dependency_profile.update(result.dependency_profile)
        for key, count in result.api_features.items():
            api_features[key] = api_features.get(key, 0) + count

    return AstExtractionResult(
        code_ops=sorted(code_ops),
        source_tags=sorted(source_tags),
        sink_tags=sorted(sink_tags),
        bridge_tags=sorted(bridge_tags),
        dependency_profile=sorted(dependency_profile),
        api_features=api_features,
    )


def build_skill_fingerprint(
    artifact: SkillArtifact,
    *,
    embedder: DocumentEmbedder | None = None,
    cluster_id: str,
    category: str,
    heat_tier: str,
    doc_embedding: list[float] | None = None,
    embedding_model: str = "",
) -> SkillFingerprintRecord:
    ast_result = extract_skill_ast_features(artifact)

    if doc_embedding is None:
        if embedder is None:
            raise ValueError("either embedder or doc_embedding must be provided")
        doc_embedding = embedder.encode(artifact.skill_doc)
        embedding_model = embedder.model_name


    attack_families: set[str] = set()
    if "credentials" in ast_result.source_tags and "network_egress" in ast_result.sink_tags:
        attack_families.add("P6")

    return SkillFingerprintRecord(
        skill_id=artifact.skill_id,
        skill_path=artifact.skill_root,
        schema_version="subgraph.v1",
        cluster_id=cluster_id,
        category=category,
        heat_tier=heat_tier,
        language=str(artifact.metadata.get("language", "text")),
        dependency_ecosystem=[],
        metadata=dict(artifact.metadata),
        doc_embedding_ref=None,
        doc_embedding=doc_embedding,
        embedding_model=embedding_model,
        code_ops=ast_result.code_ops,
        dependency_profile=ast_result.dependency_profile,
        source_tags=ast_result.source_tags,
        sink_tags=ast_result.sink_tags,
        bridge_tags=ast_result.bridge_tags,
        attack_family_candidates=sorted(attack_families),
        selection_reason="fingerprint_extracted",
        api_features=ast_result.api_features,
    )


def enrich_tags_from_declared_capabilities(
    record: SkillFingerprintRecord,
) -> SkillFingerprintRecord:
    """Fallback tag derivation from declared_capabilities for non-Python skills.

    Only applies when the record has NO source_tags AND NO sink_tags
    (i.e. the AST pipeline found nothing).  Declared capabilities come
    from SKILL.md keyword matching done by the skill_loader.
    """
    if record.source_tags or record.sink_tags:
        return record

    declared = record.metadata.get("declared_capabilities")
    if not isinstance(declared, list) or not declared:
        return record

    source_tags: set[str] = set()
    sink_tags: set[str] = set()

    for cap in declared:
        cap_str = str(cap)
        if cap_str in _DECLARED_SOURCE_MAP:
            source_tags.add(_DECLARED_SOURCE_MAP[cap_str])
        if cap_str in _DECLARED_SINK_MAP:
            sink_tags.add(_DECLARED_SINK_MAP[cap_str])
        if cap_str in _DECLARED_DUAL_MAP:
            src, snk = _DECLARED_DUAL_MAP[cap_str]
            source_tags.add(src)
            sink_tags.add(snk)

    if not source_tags and not sink_tags:
        return record

    record.source_tags = sorted(source_tags)
    record.sink_tags = sorted(sink_tags)
    record.selection_reason = "declared_capability_fallback"
    return record
