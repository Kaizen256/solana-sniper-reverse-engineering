#!/usr/bin/env python3
"""Execute and save the submission notebook with IPython display semantics.

The notebook contains no magics or shell escapes. Running it in one IPython namespace
keeps the local gate lightweight while preserving rich table and embedded PNG outputs
exactly where a judge expects to see them.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import matplotlib
import nbformat


matplotlib.use("Agg")
os.environ.setdefault("IPYTHONDIR", "/tmp/solana-sniper-ipython")

from IPython.core.interactiveshell import InteractiveShell  # noqa: E402
from IPython.utils.capture import capture_output  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook", type=Path)
    args = parser.parse_args()
    notebook_path = args.notebook.resolve()
    notebook = nbformat.read(notebook_path, as_version=4)
    shell = InteractiveShell.instance()
    executed = 0
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != "code":
            continue
        cell.outputs = []
        executed += 1
        cell.execution_count = executed
        with capture_output(display=True) as captured:
            result = shell.run_cell(cell.source, store_history=False)
        if result.error_before_exec is not None or result.error_in_exec is not None:
            if captured.stdout:
                sys.stdout.write(captured.stdout)
            if captured.stderr:
                sys.stderr.write(captured.stderr)
            result.raise_error()
        if captured.stdout:
            cell.outputs.append(
                nbformat.v4.new_output(
                    output_type="stream", name="stdout", text=captured.stdout
                )
            )
        if captured.stderr:
            cell.outputs.append(
                nbformat.v4.new_output(
                    output_type="stream", name="stderr", text=captured.stderr
                )
            )
        for rich_output in captured.outputs:
            cell.outputs.append(
                nbformat.v4.new_output(
                    output_type="display_data",
                    data=rich_output.data,
                    metadata=rich_output.metadata,
                )
            )
    nbformat.write(notebook, notebook_path)
    print(
        json.dumps(
            {
                "notebook": str(args.notebook),
                "executed_code_cells": executed,
                "saved_outputs": True,
            }
        )
    )


if __name__ == "__main__":
    main()
