"""Verify split boundaries, strict causal histories, targets, and saved outputs."""
import argparse
from pathlib import Path

import numpy as np

from .common import ROOT, read_json, sha256, write_json
from .data import load_split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--require-results", action="store_true")
    args = parser.parse_args()
    root = args.root
    directory = root / "data/processed"
    audit = read_json(directory / "audit.json")
    catalog = read_json(directory / "catalog.json")
    report = {"checks": [], "files": {}}
    vocab = len(catalog["counts"])
    bounds = [(0, audit["train_end_exclusive"]),
              (audit["train_end_exclusive"], audit["validation_end_exclusive"]),
              (audit["validation_end_exclusive"], audit["timestamp_max"] + 1)]
    for split_name, (low, high) in zip(["train", "validation", "test"], bounds):
        split = load_split(directory, split_name)
        assert ((split["timestamp"] >= low) & (split["timestamp"] < high)).all()
        assert (split["history_last_timestamp"] < split["timestamp"]).all()
        assert ((split["target"] >= (2 if split_name == "train" else 1)) & (split["target"] < vocab)).all()
        assert ((split["x"] >= 0) & (split["x"] < vocab)).all()
        assert (np.sum(split["x"] != 0, axis=1) >= audit["config"]["min_history"]).all()
        assert ((split["reward"] >= 0) & (split["reward"] <= 1)).all()
        if split_name != "train":
            assert len(np.unique(split["uid"])) == len(split["uid"])
        report["checks"].append({"split": split_name, "examples": len(split["x"]), "chronology": "passed", "causal_context": "passed"})
        report["files"][f"{split_name}.npz"] = sha256(directory / f"{split_name}.npz")
    if args.require_results:
        cfg = read_json(root / "configs/main.json")
        for seed in cfg["seeds"]:
            summary = read_json(root / f"results/seed_{seed}/summary.json")
            assert set(summary) == {"base", *cfg["variants"]}
            for variant in summary:
                assert (root / f"results/seed_{seed}/{variant}.pt").is_file()
                for split_name in ["validation", "test"]:
                    values = summary[variant][split_name]
                    assert values["queries"] == audit[f"{split_name}_examples"]
                    assert all(np.isfinite(v) for v in values.values() if isinstance(v, (int, float)))
        report["checks"].append({"results": "all requested seeds and variants present"})
    write_json(root / "results/verification.json", report)
    print(report)


if __name__ == "__main__":
    main()
