#!/usr/bin/env python3
"""Execute the training notebook cell by cell while streaming progress to the terminal."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path


def terminal_display(value) -> None:
    if hasattr(value, "to_string"):
        print(value.to_string())
    else:
        print(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--notebook",
        type=Path,
        default=Path("notebooks/meat_freshness_9class_multirun.ipynb"),
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run the complete four-model, five-seed experiment instead of the safe pilot.",
    )
    args = parser.parse_args()
    if args.full:
        os.environ["MEAT_FRESHNESS_FULL_RUN"] = "1"
    else:
        os.environ.pop("MEAT_FRESHNESS_FULL_RUN", None)

    default_notebook = Path("notebooks/meat_freshness_9class_multirun.ipynb").resolve()
    requested_notebook = args.notebook.resolve()
    if requested_notebook == default_notebook:
        generator = Path(__file__).resolve().with_name("generate_9class_multirun_notebook.py")
        subprocess.run([sys.executable, str(generator)], check=True)
        print("Notebook refreshed from the verified generator.", flush=True)

    notebook_path = args.notebook.resolve()
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    namespace = {
        "__name__": "__notebook__",
        "__file__": str(notebook_path),
        "display": terminal_display,
    }

    code_cells = [
        (index, "".join(cell.get("source", [])))
        for index, cell in enumerate(notebook["cells"])
        if cell.get("cell_type") == "code"
    ]
    print(f"Notebook: {notebook_path}", flush=True)
    print(f"Requested mode: {'full' if args.full else 'pilot'}", flush=True)
    print(f"Executable code cells: {len(code_cells)}", flush=True)

    for position, (cell_index, source) in enumerate(code_cells, start=1):
        print(f"\n--- Running code cell {position}/{len(code_cells)} (notebook cell {cell_index}) ---", flush=True)
        try:
            exec(compile(source, f"{notebook_path.name}:cell_{cell_index}", "exec"), namespace)
        except KeyboardInterrupt:
            print("\nExecution interrupted. Completed epoch checkpoints remain available.", flush=True)
            raise
        except Exception:
            print(f"\nFAILED at notebook cell {cell_index}", flush=True)
            traceback.print_exc()
            sys.exit(1)

    print("\nNotebook execution completed successfully.", flush=True)


if __name__ == "__main__":
    main()
