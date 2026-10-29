import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from shared.loaders import load_repository_slice, load_skill_artifact


class LoaderTests(unittest.TestCase):
    def test_skill_loader_extracts_capabilities_and_guidance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            skill_root = Path(tmp) / "doc-summarizer"
            scripts_dir = skill_root / "scripts"
            scripts_dir.mkdir(parents=True)
            (skill_root / "SKILL.md").write_text(
                textwrap.dedent(
                    """
                    # Doc Summarizer
                    This skill reads repository files and summarizes results.
                    As a best practice, always begin by reviewing the README.
                    """
                ).strip(),
                encoding="utf-8",
            )
            (scripts_dir / "main.py").write_text(
                "import pathlib\nprint('summary')\n",
                encoding="utf-8",
            )

            artifact = load_skill_artifact(skill_root)

        self.assertEqual(artifact.skill_id, "doc-summarizer")
        self.assertEqual(artifact.entry_script, "scripts/main.py")
        self.assertIn("file_read", artifact.declared_capabilities)
        self.assertIn("authority_framing", artifact.guidance_cues)

    def test_repo_loader_extracts_inventory_and_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp) / "repo"
            (repo_root / "docs").mkdir(parents=True)
            (repo_root / "README.md").write_text(
                "Repository for document summaries and safe reporting.",
                encoding="utf-8",
            )
            (repo_root / "docs" / "guide.md").write_text(
                "Usage guide", encoding="utf-8"
            )

            repo = load_repository_slice(repo_root)

        self.assertEqual(repo.repo_id, "repo")
        self.assertIn("README.md", repo.file_inventory)
        self.assertIn("repository", repo.theme_tokens)


if __name__ == "__main__":
    unittest.main()
