#!/usr/bin/env python3
"""Inspect the DARTIS tab-separated metadata schema without changing it."""
import argparse
import csv
from pathlib import Path

from _common import project_root


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=project_root())
    args = parser.parse_args()
    path = args.root / "DARTIS_2019.tab"
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    print(f"rows: {len(rows)}")
    print("columns:", ", ".join(rows[0].keys()) if rows else "(none)")
    for column in rows[0] if rows else []:
        print(f"{column}: {len({r[column] for r in rows if r[column]})} non-empty unique values")


if __name__ == "__main__":
    main()
