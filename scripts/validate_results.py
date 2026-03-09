"""Validate new result JSONs against reference tolerances."""

from __future__ import annotations
import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--acc_tol", type=float, default=0.005)
    args = ap.parse_args()

    ref = json.loads(Path(args.reference).read_text())
    cand = json.loads(Path(args.candidate).read_text())

    ref_acc = float(ref["accuracy"])
    cand_acc = float(cand["accuracy"])
    delta = abs(ref_acc - cand_acc)

    print(f"reference={ref_acc:.6f} candidate={cand_acc:.6f} delta={delta:.6f}")
    if delta > args.acc_tol:
        raise SystemExit(f"FAIL: accuracy deviation {delta:.6f} > {args.acc_tol:.6f}")
    print("PASS")


if __name__ == "__main__":
    main()
