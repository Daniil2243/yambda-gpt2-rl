"""Add the supervised-continuation control to already completed primary runs.

The normal experiment CLI runs this control automatically from configs/main.json.
This command only avoids repeating the primary runs when adding a control.
"""
from .common import ROOT, read_json, seed_all, write_json, sha256
from .data import load_split
from .experiment import fit_rl, load_weights, record_evaluation
from .models import Policy, RewardModel


def main():
    cfg = read_json(ROOT / "configs/main.json")
    seed_all(17, cfg["threads"])
    data = ROOT / "data/processed"
    train, val, test = [load_split(data, s) for s in ["train", "validation", "test"]]
    catalog = read_json(data / "catalog.json")
    vocab = len(catalog["counts"])
    rm = load_weights(ROOT / "results/reward.pt", RewardModel(vocab, cfg["reward_embedding_dim"]), "cpu").eval()
    for p in rm.parameters():
        p.requires_grad_(False)
    for seed in cfg["seeds"]:
        out = ROOT / f"results/seed_{seed}"
        base = load_weights(out / "base.pt", Policy(cfg, vocab), "cpu").eval()
        policy, seconds = fit_rl(cfg, seed, base, train, val, rm, catalog, "sft_control", out, "cpu")
        summary = read_json(out / "summary.json")
        summary["sft_control"] = record_evaluation("sft_control", policy, rm, val, test, catalog, cfg, out, "cpu")
        summary["sft_control"]["train_seconds"] = seconds
        write_json(out / "summary.json", summary)
    env = read_json(ROOT / "results/environment.json")
    env["config"] = cfg
    env["config_sha256"] = sha256(ROOT / "configs/main.json")
    env["control_execution"] = "Primary variants completed first; supervised-continuation control added with identical RNG/reset protocol. Normal CLI reproduces all variants in one invocation."
    write_json(ROOT / "results/environment.json", env)


if __name__ == "__main__":
    main()
