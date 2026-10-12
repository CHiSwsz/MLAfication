# Migration to the unified MLAfication repository

The original `JT-Ushio/MHA2MLA` repository continues as `JT-Ushio/MLAfication`.
Its commit history, issues, pull requests, stars, and forks stay with that
repository. Existing checkouts can update their remote:

```bash
git remote set-url origin https://github.com/JT-Ushio/MLAfication.git
git pull --ff-only
```

Before pulling, save local edits. If you work on a historical branch, reconcile
it with the new layout before moving experimental files.

## Project locations

| Former repository | Directory in this repository | Working directory for commands |
| --- | --- | --- |
| `JT-Ushio/MHA2MLA` | `MHA2MLA/` | `cd MHA2MLA` |
| `JT-Ushio/MHA2MLA-VLM` | `MHA2MLA-VLM/` | `cd MHA2MLA-VLM` |
| `JT-Ushio/MLAfication_Temp` | `MLAfication/` | `cd MLAfication` |

Every project's tracked default-branch files were imported, including
requirements, source licenses, figures, configuration files, published rank
selection tensors, and the VLM project's modified `lmms-eval` tree. The exact
source commits are recorded in [SOURCES.json](SOURCES.json). The other source
repositories remain available with their own histories and branches.

## Running experiments after migration

- Select the project before installing dependencies. Each project retains
  its own requirements and environment instructions.
- Resolve configuration paths from the selected project directory. Existing
  local model/data paths in the original ACL configs need to be replaced on
  another machine. Saved absolute paths do not become portable through a
  repository move.
- For the VLM project, training, SVD, and evaluation launchers now resolve the
  project directory automatically. Activate the Python environment before
  invoking them. The SVD launcher uses the entry point's `--stage1_path`
  argument. Evaluation always loads the bundled modified `lmms-eval`.
- For MLAfication, the internal `mha_gqa/` and `qknorm/` layouts are preserved;
  scripts that locate neighboring files through `__file__` retain their
  original relationships.
- Model weights, datasets, generated tensors, checkpoints, and results live
  outside version control. The published small rank tensors imported from
  the ACL/AAAI repositories remain tracked.

## Checks and release metadata

Run the CPU-only repository checks from the top level:

```bash
python3 scripts/check_repository.py
```

These validate project layout, owned documentation links, Python syntax, and
shell syntax. They do not run training, download datasets, or validate GPU
numerics/performance. Paper-reproduction commands remain project-specific.

The third paper is listed as **under review**. The title and authors are included from the prepared manuscript metadata.
Add its arXiv identifier and update the citation after announcement. Update the main README, the MLAfication project README, and
`CITATIONS.bib` together; no placeholder arXiv URL is published.
