from __future__ import annotations

DEFAULT_SKILL_DOC = "SKILL.md"
DEFAULT_LABEL_FILE = "labels.json"
DEFAULT_MANIFEST_FILE = "manifest.json"
DEFAULT_ALLOWED_SCRIPT_SUFFIXES = (".py", ".sh", ".js", ".ts")
DEFAULT_README_FILES = ("README.md", "README.rst", "README.txt")

GUIDANCE_PATTERNS: dict[str, tuple[str, ...]] = {
    "autonomy_encouragement": (
        "without asking",
        "without confirmation",
        "without waiting for approval",
        "without waiting for confirmation",
        "autonomously",
        "do not ask the user",
        "do not wait for confirmation",
        "no need to confirm",
        "proceed automatically",
        "proceed directly",
    ),
    "authority_framing": (
        "best practice",
        "recommended workflow",
        "standard operating procedure",
        "must always",
        "always ensure",
    ),
    "disguised_required_step": (
        "required before continuing",
        "mandatory preparation",
        "necessary prerequisite",
        "must complete this step",
    ),
}

SENSITIVE_PATH_MARKERS = (
    ".ssh",
    ".aws",
    ".env",
    "credentials",
    "id_rsa",
    "secrets",
    "token",
    "api_key",
)

# Backward-compatible alias used by the current blue-team extractor code.
SENSITIVE_PATH_HINTS = SENSITIVE_PATH_MARKERS

NETWORK_MARKERS = (
    "http://",
    "https://",
    "requests.",
    "urllib",
    "curl ",
    "wget ",
)

CAPABILITY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "file_read": ("read", "load", "inspect", "review", "parse"),
    "file_write": ("write", "save", "modify", "update", "patch"),
    "shell_exec": ("bash", "shell", "command", "execute", "run"),
    "network_access": ("http", "https", "download", "upload", "api", "network"),
    "credential_access": ("credential", "token", "secret", "ssh", "key"),
}
