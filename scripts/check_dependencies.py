#!/usr/bin/env python3
"""Resolve a maintained Linux environment against its lock, then audit it."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    policy = json.loads((ROOT / "security/dependency-policy.json").read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", choices=policy["profiles"])
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args()
    profile = policy["profiles"][args.profile]
    locked = (ROOT / profile["lock"]).read_bytes()

    with tempfile.TemporaryDirectory() as directory:
        candidate = Path(directory) / "requirements.txt"
        # uv prefers existing pins while still checking all dependency metadata.
        candidate.write_bytes(locked)
        resolve = [
            "uv", "pip", "compile", profile["manifest"],
            "--overrides", profile["overrides"],
            "--python-version", policy["python_version"],
            "--python-platform", policy["python_platform"],
            "--no-header", "--no-annotate", "--quiet",
            "--output-file", str(candidate),
        ]
        if args.cache_dir:
            resolve.extend(["--cache-dir", str(args.cache_dir.resolve() / "uv")])
        subprocess.run(resolve, cwd=ROOT, check=True)
        if candidate.read_bytes() != locked:
            raise SystemExit(f"{profile['lock']} is out of sync with its manifest/overrides")

        report_path = Path(directory) / "audit.json"
        audit = [
            sys.executable, "-m", "pip_audit", "-r", str(candidate),
            "--no-deps", "--disable-pip", "--progress-spinner", "off",
            "--format", "json", "--output", str(report_path),
        ]
        if args.cache_dir:
            audit.extend(["--cache-dir", str(args.cache_dir.resolve() / "audit")])
        result = subprocess.run(audit, cwd=ROOT)
        if not report_path.exists():
            raise SystemExit(result.returncode or "Auditor produced no report")
        report = json.loads(report_path.read_text())
        dependencies = report["dependencies"]
        skips = {item["name"] for item in dependencies if "skip_reason" in item}
        unexpected = skips - set(profile["allowed_source_audit_skips"])
        if unexpected:
            raise SystemExit(f"Unexpected unaudited dependencies: {sorted(unexpected)}")
        vulnerabilities = sum(len(item.get("vulns", [])) for item in dependencies)
        print(f"{args.profile}: {len(dependencies)} entries, {vulnerabilities} known vulnerabilities", flush=True)
        if skips:
            print(f"Source commits outside package-version audit coverage: {', '.join(sorted(skips))}")
        if result.returncode or vulnerabilities:
            raise SystemExit(result.returncode or 1)


if __name__ == "__main__":
    main()
