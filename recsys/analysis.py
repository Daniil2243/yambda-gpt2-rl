"""Aggregate seeds and paired per-user bootstrap intervals; draw research plots."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .common import ROOT, read_json, write_json

LABELS = {"popularity": "Popularity", "recency": "Recency", "base": "GPT-2",
          "sft_control": "SFT control", "completion_weak_kl": "Completion / weak KL",
          "completion_strong_kl": "Completion / strong KL", "listening_time": "Listening time", "balanced": "Balanced"}
COLORS = {"base": "#193b59", "sft_control": "#8b98a5", "completion_weak_kl": "#ef8b2c",
          "completion_strong_kl": "#be5a27", "listening_time": "#7980c8", "balanced": "#219b8f"}


def paired_interval(difference, seed=812, draws=2000):
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(0, draws, 100):
        index = rng.integers(0, len(difference), size=(min(100, draws - len(means)), len(difference)))
        means.extend(difference[index].mean(1))
    return {"mean_delta": float(difference.mean()), "low_95": float(np.quantile(means, .025)),
            "high_95": float(np.quantile(means, .975)), "resamples": draws}


def main():
    cfg = read_json(ROOT / "configs/main.json")
    audit = read_json(ROOT / "data/processed/audit.json")
    seeds = cfg["seeds"]
    summaries = [read_json(ROOT / f"results/seed_{s}/summary.json") for s in seeds]
    variants = ["base", *cfg["variants"]]
    results = ROOT / "results"
    figures = results / "figures"
    figures.mkdir(exist_ok=True)
    aggregate = {"seeds": seeds, "models": {}, "paired_bootstrap_vs_base": {},
                 "paired_bootstrap_vs_sft_control": {},
                 "uncertainty_scope": "Bootstrap resamples test users after averaging each query over the three policy seeds. This is conditional on the shared dataset, catalog, and single frozen reward model; it is not a causal confidence interval or a full training-uncertainty interval."}
    for variant in variants:
        metrics = {}
        for key in summaries[0][variant]["test"]:
            values = np.array([summary[variant]["test"][key] for summary in summaries])
            metrics[key] = {"mean": float(values.mean()), "std": float(values.std(ddof=1)), "values": values.tolist()}
        rollout = {}
        for key in ["discounted_model_return", "within_slate_duplicate_fraction", "original_history_repeat_fraction", "unique_generated_items"]:
            values = np.array([summary[variant]["rollout"][key] for summary in summaries])
            rollout[key] = {"mean": float(values.mean()), "std": float(values.std(ddof=1)), "values": values.tolist()}
        aggregate["models"][variant] = {"test": metrics, "rollout": rollout,
                                       "training_seconds": [summary[variant]["train_seconds"] for summary in summaries]}
    for variant in ["popularity", "recency"]:
        metrics = read_json(results / f"{variant}_evaluation.json")["test"]
        aggregate["models"][variant] = {"test": {k: {"mean": v, "std": 0, "values": [v]} for k, v in metrics.items()}}
    for control, output in [("base", "paired_bootstrap_vs_base"), ("sft_control", "paired_bootstrap_vs_sft_control")]:
        for variant in variants:
            if variant == control:
                continue
            pairs = {}
            for key in ["recall@10", "ndcg@10", "predicted_completion@10", "sampled_completion", "history_repeat@10", "novelty@10"]:
                a, b = [], []
                for seed in seeds:
                    with np.load(results / f"seed_{seed}/{variant}_test_per_query.npz") as p:
                        a.append(p[key])
                    with np.load(results / f"seed_{seed}/{control}_test_per_query.npz") as p:
                        b.append(p[key])
                difference = np.mean(a, 0) - np.mean(b, 0)
                pairs[key] = paired_interval(difference)
            aggregate[output][variant] = pairs
    write_json(results / "aggregate.json", aggregate)
    columns = ["model", "recall@10", "ndcg@10", "catalog_recall@10", "catalog_ndcg@10", "predicted_completion@10",
               "history_repeat@10", "catalog_coverage@10", "top1_max_share", "novelty@10", "mean_duration@10_seconds"]
    with (results / "metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, columns)
        writer.writeheader()
        for variant in ["popularity", "recency", *variants]:
            writer.writerow({"model": variant, **{k: aggregate["models"][variant]["test"][k]["mean"] for k in columns[1:]}})
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "figure.facecolor": "white"})
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), layout="constrained")
    names = ["base", "sft_control", "completion_weak_kl", "completion_strong_kl", "listening_time", "balanced"]
    for ax, metric, title in zip(axes, ["ndcg@10", "predicted_completion@10"], ["Relevance: NDCG@10", "Frozen-model completion@10"]):
        y = [aggregate["models"][n]["test"][metric]["mean"] for n in names]
        errors = [aggregate["models"][n]["test"][metric]["std"] for n in names]
        ax.barh(np.arange(len(names)), y, xerr=errors, color=[COLORS[n] for n in names], capsize=3)
        ax.set_yticks(np.arange(len(names)), [LABELS[n] for n in names], fontsize=8)
        ax.invert_yaxis()
        ax.set_title(title)
        ax.grid(axis="x", alpha=.15)
    fig.savefig(figures / "main_comparison.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.5, 3.8), layout="constrained")
    for name in names:
        means = np.mean([s[name]["rollout"]["mean_predicted_completion_by_step"] for s in summaries], axis=0)
        ax.plot(range(1, cfg["rollout_horizon"] + 1), means, marker="o", color=COLORS[name], label=LABELS[name])
    ax.set(xlabel="Generated step (hypothetical consumed item)", ylabel="Frozen-model completion", title="Generated suffixes: reward-model diagnostics")
    ax.legend(ncol=2, fontsize=8)
    ax.set_xticks(range(1, cfg["rollout_horizon"] + 1))
    ax.grid(alpha=.15)
    fig.savefig(figures / "rollout.png", dpi=180)
    plt.close(fig)
    rm = read_json(results / "reward_metrics.json")["splits"]["test"]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), layout="constrained")
    keys = ["global_mean", "item_mean_shrunk", "neural", "neural_shuffled_history"]
    axes[0].barh(["Global mean", "Shrunk item mean", "History + item model", "Shuffled histories"], [rm[k]["mse"] for k in keys], color=["#8b98a5", "#219b8f", "#193b59", "#ef8b2c"])
    axes[0].set(title="Response model: held-out MSE", xlabel="MSE (lower is better)")
    axes[0].invert_yaxis()
    bins = rm["calibration"]
    axes[1].plot([0, 1], [0, 1], "--", color="#9cabb5")
    axes[1].scatter([b["predicted"] for b in bins], [b["observed"] for b in bins], s=[min(150, 10 + b["n"] / 5) for b in bins], color="#193b59")
    axes[1].set(title="Calibration on logged test events", xlabel="Mean prediction", ylabel="Mean observed completion", xlim=(0, 1), ylim=(0, 1))
    fig.savefig(figures / "reward_validation.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.7), layout="constrained")
    for name in names:
        m = aggregate["models"][name]["test"]
        axes[0].scatter(m["history_repeat@10"]["mean"], m["predicted_completion@10"]["mean"], color=COLORS[name], s=65, label=LABELS[name])
        axes[1].scatter(m["catalog_coverage@10"]["mean"], m["ndcg@10"]["mean"], color=COLORS[name], s=65)
    axes[0].set(xlabel="Fraction of Top-10 already in last 32 events", ylabel="Predicted completion@10", title="Reward and recent-history repetition")
    axes[1].set(xlabel="Catalog coverage@10", ylabel="NDCG@10", title="Coverage and relevance")
    axes[0].legend(fontsize=7)
    fig.savefig(figures / "behavior.png", dpi=180)
    plt.close(fig)
    print("Aggregate metrics, paired intervals, CSV, and four figures generated.")


if __name__ == "__main__":
    main()
