#!/usr/bin/env python3
"""Check the consolidated repository without importing ML dependencies."""

import ast
import json
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import unquote, urlsplit
import warnings

ROOT = Path(__file__).resolve().parents[1]
PROJECTS = (
    "MHA2MLA",
    "MHA2MLA-VLM",
    "MLAfication",
)
errors = []

for project in PROJECTS:
    for name in ("README.md", "LICENSE", "requirements.txt"):
        if not (ROOT / project / name).is_file():
            errors.append(f"Missing project file: {project}/{name}")

sources = json.loads((ROOT / "docs/SOURCES.json").read_text())
if {item["project"] for item in sources["projects"]} != set(PROJECTS):
    errors.append("Source manifest does not match project directories")
for item in sources["projects"]:
    if not re.fullmatch(r"[0-9a-f]{40}", item["source_commit"]):
        errors.append(f"Invalid source commit: {item['project']}")

# Validate the maintained guides; bundled third-party docs keep upstream links.
guides = [ROOT / "README.md", ROOT / "THIRD_PARTY_NOTICES.md"]
guides += list((ROOT / "docs").glob("*.md"))
guides += [ROOT / project / "README.md" for project in PROJECTS]
guides += [ROOT / "MLAfication" / name / "README.md" for name in ("mha_gqa", "qknorm")]
guides += [ROOT / "MLAfication/THIRD_PARTY_NOTICES.md"]
link_count = 0
for guide in guides:
    for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", guide.read_text()):
        target = target.strip().split(" ", 1)[0].strip("<>")
        parts = urlsplit(target)
        if parts.scheme or parts.netloc or not parts.path:
            continue
        link_count += 1
        resolved = (guide.parent / unquote(parts.path)).resolve()
        if not resolved.is_relative_to(ROOT) or not resolved.exists():
            errors.append(f"Broken local link in {guide.relative_to(ROOT)}: {target}")

python_count = shell_count = 0
with warnings.catch_warnings():
    warnings.simplefilter("ignore", SyntaxWarning)
    for project in PROJECTS:
        for path in (ROOT / project).rglob("*.py"):
            python_count += 1
            try:
                ast.parse(path.read_bytes(), filename=str(path.relative_to(ROOT)))
            except (SyntaxError, UnicodeError) as exc:
                errors.append(str(exc))
        # Our launchers only; bundled eval framework scripts retain upstream behavior.
        for path in (ROOT / project).rglob("*.sh"):
            if "lmms-eval" in path.parts:
                continue
            shell_count += 1
            result = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
            if result.returncode:
                errors.append(f"{path.relative_to(ROOT)}: {result.stderr.strip()}")

if errors:
    print("\n".join(errors), file=sys.stderr)
    sys.exit(1)
print(f"OK: {len(PROJECTS)} projects, {python_count} Python files, "
      f"{shell_count} shell scripts, {link_count} local documentation links")
print("Static checks only; no GPU training or evaluation was run.")
