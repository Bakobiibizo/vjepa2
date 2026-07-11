# Bakobiibizo integration fork

This repository is retained as a deployable integration of Meta's V-JEPA 2 for video representation and world-model experiments. Model architecture and research changes continue to come from `facebookresearch/vjepa2`. This fork owns the operational layer needed to run that work reproducibly.

## Fork policy

- Keep `upstream/main` configured and prefer clean upstream merges.
- Do not modify model semantics unless independently tested and documented.
- Keep checkpoints and datasets outside Git. Downloads must be explicit.
- CPU supports environment checks and small smoke tests. Practical inference and training require an accelerator; CUDA is the validated target.
- ARM64 is supported when PyTorch provides a wheel for the selected Python/CUDA combination.

When this policy was introduced, `main` and `upstream/main` had no divergent commits. Integration files are isolated under `scripts/`, `tests/integration/`, and this document.

## Updating from upstream

```bash
git fetch upstream
git log --left-right --oneline main...upstream/main
git merge --ff-only upstream/main
python scripts/check_environment.py --json
python -m unittest discover -s tests/integration -v
```

If fast-forward fails, review and merge explicitly; never discard integration commits with a forced reset.
