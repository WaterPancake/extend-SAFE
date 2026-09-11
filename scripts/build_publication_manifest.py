"""Build a deterministic SHA-256 manifest for the public audit surface."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.score_bundle import sha256_file


def relative_hashes(paths: list[Path]) -> dict[str, str]:
    return {
        path.relative_to(REPO_ROOT).as_posix(): sha256_file(path)
        for path in sorted(paths)
        if path.is_file()
    }


def git_revision() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def build_manifest(output: Path) -> dict:
    result_root = REPO_ROOT / "docs" / "results_audit"
    result_files = [
        path
        for path in result_root.rglob("*")
        if path.is_file() and path.resolve() != output.resolve()
    ]
    code_files = [
        path
        for root in ("data", "models", "scripts", "tests")
        for path in (REPO_ROOT / root).rglob("*.py")
    ]
    code_files.extend((REPO_ROOT / ".github" / "workflows").glob("*.yml"))
    code_files.extend((REPO_ROOT / ".github" / "workflows").glob("*.yaml"))
    code_files.extend(
        REPO_ROOT / name
        for name in (".gitignore", "pyproject.toml", "uv.lock")
        if (REPO_ROOT / name).is_file()
    )
    report_files = list((REPO_ROOT / "docs").glob("*"))
    report_files.extend(
        [
            REPO_ROOT / "README.md",
            REPO_ROOT / "DATA.md",
            REPO_ROOT / "THIRD_PARTY.md",
            REPO_ROOT / "LICENSE",
            REPO_ROOT / "CITATION.cff",
        ]
    )
    manifest = {
        "schema_version": 1,
        "release": "1.0.0",
        "source_revision_before_publication_commit": git_revision(),
        "integrity_note": (
            "Per-file hashes are authoritative for the publication commit; the "
            "revision records the experiment base before closeout files were committed."
        ),
        "reports": relative_hashes(report_files),
        "result_artifacts": relative_hashes(result_files),
        "analysis_code_and_environment": relative_hashes(code_files),
        "generator": "uv run python scripts/build_publication_manifest.py",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "docs" / "results_audit" / "publication_manifest.json",
    )
    args = parser.parse_args()
    manifest = build_manifest(args.output.resolve())
    count = sum(
        len(manifest[key])
        for key in ("reports", "result_artifacts", "analysis_code_and_environment")
    )
    print(f"wrote {args.output} with {count} file hashes")


if __name__ == "__main__":
    main()
