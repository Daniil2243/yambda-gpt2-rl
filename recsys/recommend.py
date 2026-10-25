"""Inspect recommendations from a trained checkpoint."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .common import ROOT, read_json, seed_all
from .data import load_split
from .experiment import load_weights
from .models import Policy, RewardModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "results/seed_17/base.pt")
    parser.add_argument("--history", type=int, nargs="+", help="Raw track IDs in chronological order")
    parser.add_argument("--demo", type=int, default=0, help="Prepared test query index, if --history is omitted")
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()
    needed = [args.checkpoint, ROOT / "results/reward.pt", ROOT / "data/processed/catalog.json"]
    if not args.history:
        needed.append(ROOT / "data/processed/test.npz")
    if any(not path.is_file() for path in needed):
        parser.error("Prepared data and trained weights are required. Run python run_all.py first.")
    cfg = read_json(ROOT / "configs/main.json")
    seed_all(17, cfg["threads"])
    catalog = read_json(ROOT / "data/processed/catalog.json")
    vocab = len(catalog["counts"])
    if not 1 <= args.k <= vocab - 2:
        parser.error("k must be between 1 and catalog size")
    if args.history:
        mapping = {raw: token for token, raw in enumerate(catalog["item_ids"]) if token >= 2}
        history = [mapping.get(raw, 1) for raw in args.history][-cfg["context_length"]:]
        x = np.zeros((1, cfg["context_length"]), dtype=np.int64)
        x[0, :len(history)] = history
        uid = None
    else:
        split = load_split(ROOT / "data/processed", "test")
        if not 0 <= args.demo < len(split["x"]):
            parser.error("demo index out of range")
        x = split["x"][args.demo:args.demo + 1]
        uid = int(split["uid"][args.demo])
    x = torch.tensor(x, dtype=torch.long)
    policy = load_weights(args.checkpoint, Policy(cfg, vocab), "cpu").eval()
    rm = load_weights(ROOT / "results/reward.pt", RewardModel(vocab, cfg["reward_embedding_dim"]), "cpu").eval()
    with torch.inference_mode():
        probs = policy(x).softmax(1)
        probability, indices = probs.topk(args.k, 1)
        tokens = indices + 2
        rewards = rm(x, tokens)
    result = {"uid": uid, "checkpoint": str(args.checkpoint),
              "recommendations": [{"item_id": catalog["item_ids"][token],
                                   "probability": float(p), "predicted_completion": float(r),
                                   "in_last_history": token in x[0].tolist()}
                                  for token, p, r in zip(tokens[0].tolist(), probability[0], rewards[0])]}
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
