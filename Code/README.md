# IDS 26: Assignment Part 1
This directory provides the main artifacts for the first part of the assignment:

- `ids26-assignment-p1.pdf`: The task sheet.
- `data/*.csv`: The datasets.
- `notebook-student.ipynb`: The notebook scaffold.
- `helpers.py`: Additional functionality used in the notebook. Keep it in the same directory.
- `.python-version`, `pyproject.toml`, `uv.lock`: Files related to `uv`.

## Setup

**Course Environment**: Upload the directory contents (data, notebook, and `helpers.py`) and edit the notebook as needed.

**UV**: This directory includes a basic `uv`-managed project for execution. If `uv` is installed, run:

1. `uv sync`
2. `uv run --with jupyter jupyter lab`

