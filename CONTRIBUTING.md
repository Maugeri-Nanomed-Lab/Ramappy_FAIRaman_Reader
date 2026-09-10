# Contributing

## Design rules

- UI modules must not implement scientific algorithms.
- Analysis modules must not import Flet.
- Export modules must not depend on Flet controls.
- New advanced analyses return `AnalysisResult` and are registered in `analysis/registry.py`.
- Long-running processing must remain cancellable and must report progress through callbacks.
- Do not silently transform scientific data to satisfy algorithm constraints; surface the requirement to the user.

## Before committing

```bash
ruff check src tests
pytest
python -m compileall -q src
```
