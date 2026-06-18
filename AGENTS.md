# Repository Guidelines

## Project Structure & Module Organization
- `assets/` holds figures used in the top-level `README.md`.
- `Infinity/` contains the FastVAR integration for the Infinity VAR backbone (`inference.py`, `infinity/`, `evaluation/`, `tools/`).
- `HART/` contains the FastVAR integration for the HART backbone (`inference.py`, `hart/`, `evaluation/`).
- Subproject-specific evaluation instructions live in `Infinity/evaluation/README.md` and `HART/evaluation/README.md`.

## Build, Test, and Development Commands
- Run Infinity inference from the repo root:
  ```bash
  cd Infinity
  python inference.py
  ```
- Run HART inference (requires model paths):
  ```bash
  cd HART
  python inference.py --model_path /path/to/model --text_model_path /path/to/Qwen2 \
    --prompt "YOUR_PROMPT" --sample_folder_dir /path/to/save_dir
  ```
- Dependencies are defined per subproject (`Infinity/requirements.txt`, `HART/pyproject.toml`). Use the matching toolchain for each subfolder.

## Coding Style & Naming Conventions
- Code is primarily Python; follow existing style in each subproject.
- Use 4-space indentation, `snake_case` for functions/variables, and `CamelCase` for classes.
- HART includes tooling for formatting (`black`) and import sorting (`isort` with the Black profile). If you touch HART, format accordingly.

## Testing Guidelines
- There is no standalone unit-test suite in this repository.
- Evaluation and benchmarking scripts live under `Infinity/evaluation/` and `HART/evaluation/`; follow the README in those folders for dataset-specific runs.

## Commit & Pull Request Guidelines
- Git history does not show a formal commit convention; use short, imperative summaries (e.g., "Add HART eval flag handling").
- For PRs, include: a brief summary, the subproject touched (`Infinity` or `HART`), and any reproducibility notes (model checkpoints, datasets, or commands used). Add before/after metrics or sample outputs when results change.

## Configuration & Assets
- Model checkpoints, datasets, and third-party backbones are external to the repo; document any required paths in your PR or issue.
- Keep large binary assets out of the repo unless explicitly requested; prefer links or generation scripts.
