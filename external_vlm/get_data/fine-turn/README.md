# Ball-in-Cup VLM Fine-Tune

This subproject trains lightweight VLM heads and optional LoRA adapters on the
paired Ball-in-Cup dataset produced by the parent `get_data` project.

The intended training signal is narrow:

- background pairs encourage task embeddings to ignore DAVIS background changes;
- interaction pairs provide weak relation sensitivity for real state changes;
- the VLM exports semantic candidates (`q_sem`) for a later DreamerV3 verifier;
- the VLM is not trained to replace the action-conditioned world model.

All runtime paths are resolved from CLI flags, environment variables, YAML
config, or safe paths relative to this source tree.

Quick local checks:

```bash
python scripts/inspect_existing_code.py --dry-run
python scripts/inspect_dataset.py --dry-run
pytest -q
```

Server usage is documented in `docs/server_commands.md`.

