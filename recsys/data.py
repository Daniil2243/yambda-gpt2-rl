"""Streaming audit, global chronological split, and causal prefix extraction.

PAD=0, OOV=1, catalog items=2..V+1. OOV events stay in histories; we never
delete them to manufacture adjacent interactions. Targets outside the catalog
remain in validation/test and count as retrieval misses.
"""
import argparse
import time
import urllib.request
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from .common import ROOT, read_json, sha256, write_json

DATA_URL = "https://huggingface.co/datasets/yandex/yambda/resolve/main/sequential/50m/listens.parquet"
DATA_SHA256 = "c07db8326739b23fd4fb0a25037304e78b2e916320ccf1b01dc904d2b1b376c7"
COLUMNS = ["timestamp", "item_id", "played_ratio_pct", "track_length_seconds", "is_organic"]


def download(path):
    path = Path(path)
    if path.exists() and sha256(path) == DATA_SHA256:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial")
    with urllib.request.urlopen(DATA_URL, timeout=120) as r, temporary.open("wb") as f:
        while chunk := r.read(8 * 1024 * 1024):
            f.write(chunk)
    if sha256(temporary) != DATA_SHA256:
        raise ValueError("Dataset checksum mismatch; source changed or download incomplete")
    temporary.replace(path)


def iter_users(path):
    """Read list arrays as NumPy views, without building millions of Python ints."""
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=128):
        scalar = batch.column(batch.schema.get_field_index("uid")).to_numpy()
        arrays = {}
        for name in COLUMNS:
            arr = batch.column(batch.schema.get_field_index(name))
            if arr.null_count or arr.values.null_count:
                raise ValueError(f"Unexpected nulls in {name}")
            arrays[name] = (arr.offsets.to_numpy(), arr.values.to_numpy())
        for i, uid in enumerate(scalar):
            row = {"uid": int(uid)}
            for name, (offsets, values) in arrays.items():
                row[name] = values[offsets[i]:offsets[i + 1]]
            yield row


def make_context(tokens, timestamps, position, length):
    # All events tied with the target time are withheld: their true ordering is
    # not observable at the timestamp resolution supplied by the dataset.
    end = int(np.searchsorted(timestamps, timestamps[position], side="left"))
    history = tokens[max(0, end - length):end]
    x = np.zeros(length, dtype=np.int32)
    x[:len(history)] = history
    return x, end


def prepare(raw, out, cfg):
    start = time.time()
    raw, out = Path(raw), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    digest = sha256(raw)
    if digest != DATA_SHA256:
        raise ValueError("This experiment requires the pinned listens.parquet checksum")
    audit = {"users": 0, "events": 0, "unsorted_users": 0,
             "ratio_above_100": 0, "equal_adjacent_timestamps": 0,
             "zero_durations": 0, "negative_or_invalid_organic": 0}
    lo, hi, max_item = 2**32, 0, 0
    uids = set()
    for row in iter_users(raw):
        ts, ids = row["timestamp"], row["item_id"]
        if len({len(row[k]) for k in COLUMNS}) != 1:
            raise ValueError("Misaligned per-user lists")
        if row["uid"] in uids:
            raise ValueError("Duplicate user row")
        uids.add(row["uid"])
        if not len(ts):
            continue
        audit["users"] += 1
        audit["events"] += len(ts)
        audit["unsorted_users"] += int(np.any(ts[1:] < ts[:-1]))
        audit["equal_adjacent_timestamps"] += int((ts[1:] == ts[:-1]).sum())
        audit["ratio_above_100"] += int((row["played_ratio_pct"] > 100).sum())
        audit["zero_durations"] += int((row["track_length_seconds"] == 0).sum())
        audit["negative_or_invalid_organic"] += int((row["is_organic"] > 1).sum())
        lo, hi = min(lo, int(ts.min())), max(hi, int(ts.max()))
        max_item = max(max_item, int(ids.max()))
    if audit["unsorted_users"]:
        raise ValueError("Timestamps must be ordered")
    cut_train = int(lo + (hi - lo) * cfg["train_fraction_time"])
    cut_val = int(lo + (hi - lo) * (cfg["train_fraction_time"] + cfg["validation_fraction_time"]))
    counts = np.zeros(max_item + 1, dtype=np.int64)
    for row in iter_users(raw):
        ids = row["item_id"][row["timestamp"] < cut_train]
        np.add.at(counts, ids, 1)
    available = np.flatnonzero(counts)
    order = np.lexsort((available, -counts[available]))
    catalog = available[order[:cfg["catalog_size"]]]
    lookup = np.ones(max_item + 1, dtype=np.int32)
    lookup[catalog] = np.arange(2, len(catalog) + 2)
    vocab = len(catalog) + 2
    count_catalog = np.r_[0, 0, counts[catalog]].astype(np.int64)
    reward_sum, duration_sum = np.zeros(vocab), np.zeros(vocab)
    event_counts, known_counts = np.zeros(3, np.int64), np.zeros(3, np.int64)
    rng = np.random.default_rng(cfg["data_seed"])
    samples = {name: [] for name in ["train", "validation", "test"]}
    length = cfg["context_length"]
    for row in iter_users(raw):
        ts, ids = row["timestamp"], row["item_id"]
        tokens = lookup[ids]
        ratio = np.minimum(row["played_ratio_pct"].astype(np.float32) / 100, 1)
        # Replays >100% are valid logs. Completion deliberately caps them at one.
        split = np.where(ts < cut_train, 0, np.where(ts < cut_val, 1, 2))
        for s in range(3):
            mask = split == s
            event_counts[s] += int(mask.sum())
            known_counts[s] += int(((tokens >= 2) & mask).sum())
        train_known = (split == 0) & (tokens >= 2)
        np.add.at(reward_sum, tokens[train_known], ratio[train_known])
        np.add.at(duration_sum, tokens[train_known], row["track_length_seconds"][train_known])
        first_past = np.searchsorted(ts, ts, side="left")
        for s, name in enumerate(samples):
            eligible = (split == s) & (first_past >= cfg["min_history"])
            if s == 0:
                eligible &= tokens >= 2
            positions = np.flatnonzero(eligible)
            number = min(len(positions), cfg["train_examples_per_user"] if s == 0 else 1)
            if not number:
                continue
            positions = rng.choice(positions, number, replace=False)
            for position in positions:
                x, end = make_context(tokens, ts, position, length)
                assert ts[end - 1] < ts[position]
                samples[name].append((x, int(tokens[position]), float(ratio[position]),
                                      row["uid"], int(ts[position]), int(ts[end - 1]),
                                      int(row["is_organic"][position])))
    for name, rows in samples.items():
        if name != "train" and len(rows) > cfg["max_eval_users"]:
            chosen = np.sort(rng.choice(len(rows), cfg["max_eval_users"], replace=False))
            rows = [rows[i] for i in chosen]
        x = np.stack([r[0] for r in rows])
        result = {"x": x, "target": np.array([r[1] for r in rows], np.int64),
                  "reward": np.array([r[2] for r in rows], np.float32),
                  "uid": np.array([r[3] for r in rows], np.int64),
                  "timestamp": np.array([r[4] for r in rows], np.int64),
                  "history_last_timestamp": np.array([r[5] for r in rows], np.int64),
                  "is_organic": np.array([r[6] for r in rows], np.uint8)}
        np.savez_compressed(out / f"{name}.npz", **result)
        audit[f"{name}_examples"] = len(rows)
        audit[f"{name}_known_targets"] = int((result["target"] >= 2).sum())
    item_mean_reward = np.divide(reward_sum, count_catalog, out=np.zeros(vocab), where=count_catalog > 0)
    item_mean_duration = np.divide(duration_sum, count_catalog, out=np.zeros(vocab), where=count_catalog > 0)
    global_reward = float(reward_sum.sum() / count_catalog.sum())
    shrunk_reward = (reward_sum + 20 * global_reward) / (count_catalog + 20)
    # Normalized self-information; fitted to train counts only.
    novelty = -np.log((count_catalog + 1) / (count_catalog.sum() + len(catalog)))
    novelty[:2] = 0
    novelty[2:] /= novelty[2:].max()
    write_json(out / "catalog.json", {"item_ids": [None, None] + catalog.tolist(),
               "counts": count_catalog.tolist(), "mean_completion": item_mean_reward.tolist(),
               "shrunk_completion": shrunk_reward.tolist(), "global_completion": global_reward,
               "mean_duration_seconds": item_mean_duration.tolist(), "novelty": novelty.tolist()})
    audit.update({"source_url": DATA_URL, "sha256": digest, "bytes": raw.stat().st_size,
                  "timestamp_min": lo, "timestamp_max": hi, "train_end_exclusive": cut_train,
                  "validation_end_exclusive": cut_val, "catalog_size": len(catalog),
                  "unique_train_items": len(available), "events_by_split": event_counts.tolist(),
                  "known_catalog_events_by_split": known_counts.tolist(),
                  "catalog_event_coverage_by_split": (known_counts / event_counts).tolist(),
                  "query_sampling": "one uniformly sampled eligible event per user in each held-out interval; max_eval_users sampled without replacement",
                  "history_policy": "last 32 strictly earlier events, OOV retained, equal-time events excluded",
                  "seconds": round(time.time() - start, 3), "config": cfg})
    write_json(out / "audit.json", audit)
    print(audit, flush=True)


def load_split(directory, name):
    with np.load(Path(directory) / f"{name}.npz", allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", type=Path, default=ROOT / "data/raw/listens.parquet")
    p.add_argument("--out", type=Path, default=ROOT / "data/processed")
    p.add_argument("--config", type=Path, default=ROOT / "configs/main.json")
    p.add_argument("--download", action="store_true")
    args = p.parse_args()
    if args.download:
        download(args.raw)
    prepare(args.raw, args.out, read_json(args.config))


if __name__ == "__main__":
    main()
