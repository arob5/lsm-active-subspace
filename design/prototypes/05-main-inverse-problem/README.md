# Prototype for design 05 (the walkthrough notebook)

Scratch code that backs the claims in [`../../05-main-inverse-problem.md`](../../05-main-inverse-problem.md).
Printed results and figures in `out/` are committed; the executed notebook is not.

| File | What it checks |
|---|---|
| `t1_sections.py` | Every planned section's code against the package, its figure, and its cost |
| `t2_execute_notebook.py` | A notebook of the opening cells executes under `nbclient` in the project environment |

```bash
uv run python design/prototypes/05-main-inverse-problem/t1_sections.py        # ~1 min
uv run --with nbformat --with nbclient --with ipykernel \
    python design/prototypes/05-main-inverse-problem/t2_execute_notebook.py   # ~15 s
```
