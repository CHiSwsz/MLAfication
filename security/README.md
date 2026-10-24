# Dependency security

The maintained environments target Python 3.11 on x86_64 Linux with glibc 2.28
or newer. Use a separate environment for each project. The project READMEs
include the installation commands and the explicit overrides needed to replace
stale LightEval/Swift dependency caps with patched releases.

`locks/` records the fully resolved package versions audited on 2026-10-24.
Each profile's manifest, overrides, and lock path are listed in
[`dependency-policy.json`](dependency-policy.json). For an exact Linux snapshot,
replace `-r requirements.txt` in the corresponding installation command with
`-r ../security/locks/<profile>-py311-linux.txt`, retaining its `--overrides`
option. These Linux locks are not portable macOS/Windows environment files.

To check a profile from the repository root:

```bash
python -m pip install uv==0.13.5 pip-audit==2.10.1
python scripts/check_dependencies.py mlafication
```

The check re-resolves the manifest against the committed pins, rejects stale
locks, and audits the complete resolved package list. CI runs it for all eight
profiles. To intentionally refresh a lock, run `uv pip compile` with the profile's
manifest and overrides, `--python-version 3.11`,
`--python-platform x86_64-manylinux_2_28`, `--no-header --no-annotate --upgrade`,
and `--output-file` pointing to the lock; then rerun the check.

Nanotron and archived LightEval Git references are fixed to immutable commits.
Package-version advisory services cannot audit those source commits as releases;
only those explicitly listed source dependencies may be skipped. Optional
`pyproject.toml` extras and arbitrary external model repositories are outside
the locked profiles.

DiskCache and standalone Outlines were unused in the primary project exports
and have been removed. The current SGLang package also pulls vulnerable
DiskCache through Outlines, so SGLang is excluded from the two default reference
environments. Its optional adapters require a separately maintained environment;
the Hugging Face and vLLM paths remain available in the maintained environments.

The evaluation response cache uses JSON in a separate SQLite table. Historical
pickle-backed tables remain untouched and are never deserialized; their entries
are recomputed. Tensor checkpoint loads explicitly use `weights_only=True`.

Validation includes NLTK's resource sandbox and evaluation primitives, JSON cache
regressions, tensor deserialization, and a tiny CPU MLA forward/backward pass.
GPU training, serving, and paper metrics have not been rerun as part of this
dependency update. Passing these checks covers the listed environments and
known advisories at audit time, not future vulnerabilities.
