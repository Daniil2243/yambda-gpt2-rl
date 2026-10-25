"""Train, align with contextual-bandit REINFORCE, and evaluate from the CLI."""
import argparse
import copy
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score
from scipy.stats import spearmanr

from .common import ROOT, append_jsonl, read_json, seed_all, sha256, write_json
from .data import load_split
from .evaluate import evaluate, rollout, tensor_x
from .models import Policy, RewardModel


def save_weights(path, model):
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, path)


def load_weights(path, model, device):
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return model


def regression_metrics(y, pred):
    labels = y >= 0.9
    return {"mse": float(mean_squared_error(y, pred)),
            "mae": float(mean_absolute_error(y, pred)),
            "spearman": float(spearmanr(y, pred).statistic) if np.std(pred) > 1e-10 else None,
            "auc_completion_ge_90pct": float(roc_auc_score(labels, pred)) if len(np.unique(labels)) > 1 else None,
            "mean_prediction": float(np.mean(pred)), "mean_observed_completion": float(np.mean(y)),
            "n": len(y)}


@torch.inference_mode()
def reward_predictions(model, split, device, shuffle_history=False):
    model.eval()
    known = np.flatnonzero(split["target"] >= 2)
    output = []
    shuffled = known.copy()
    np.random.default_rng(123).shuffle(shuffled)
    for start in range(0, len(known), 512):
        idx = known[start:start + 512]
        history_idx = shuffled[start:start + 512] if shuffle_history else idx
        x = tensor_x(split, history_idx, device)
        target = torch.tensor(split["target"][idx], device=device)
        output.append(model(x, target).cpu().numpy())
    return known, np.concatenate(output)


def fit_reward(cfg, train, val, test, catalog, out, device):
    seed_all(17, cfg["threads"])
    model = RewardModel(len(catalog["counts"]), cfg["reward_embedding_dim"]).to(device)
    path = out / "reward.pt"
    rng = np.random.default_rng(17)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["reward_lr"], weight_decay=1e-4)
    log = out / "reward_training.jsonl"
    if log.exists():
        log.unlink()
    best = float("inf")
    start = time.time()
    for step in range(1, cfg["reward_steps"] + 1):
        model.train()
        idx = rng.integers(len(train["x"]), size=256)
        x = tensor_x(train, idx, device)
        items = torch.tensor(train["target"][idx], device=device)
        y = torch.tensor(train["reward"][idx], device=device)
        prediction = model(x, items)
        loss = (prediction - y).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        if step % 200 == 0 or step == cfg["reward_steps"]:
            known, pred = reward_predictions(model, val, device)
            mse = float(np.mean((pred - val["reward"][known])**2))
            append_jsonl(log, {"step": step, "training_batch_mse": loss.item(), "validation_mse": mse})
            if mse < best:
                best = mse
                save_weights(path, model)
            print(f"reward step={step} validation MSE={mse:.5f}", flush=True)
    load_weights(path, model, device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    report = {"selection": "minimum validation MSE, evaluated every 200 updates",
              "train_seconds": round(time.time() - start, 3), "splits": {}}
    for name, split in [("validation", val), ("test", test)]:
        known, pred = reward_predictions(model, split, device)
        y = split["reward"][known]
        baseline = np.asarray(catalog["shrunk_completion"])[split["target"][known]]
        _, shuffled = reward_predictions(model, split, device, shuffle_history=True)
        result = {"neural": regression_metrics(y, pred),
                  "global_mean": regression_metrics(y, np.full(len(y), catalog["global_completion"])),
                  "item_mean_shrunk": regression_metrics(y, baseline),
                  "neural_shuffled_history": regression_metrics(y, shuffled)}
        bins = []
        for low in np.arange(0, 1, 0.1):
            mask = (pred >= low) & (pred < low + 0.1 + (1e-7 if low > .89 else 0))
            if mask.any():
                bins.append({"low": float(low), "high": float(low + .1), "n": int(mask.sum()),
                             "predicted": float(pred[mask].mean()), "observed": float(y[mask].mean())})
        result["calibration"] = bins
        for organic, label in [(0, "algorithmic"), (1, "organic")]:
            mask = split["is_organic"][known] == organic
            if mask.sum() > 3:
                result[label] = regression_metrics(y[mask], pred[mask])
        report["splits"][name] = result
        np.savez_compressed(out / f"reward_{name}_predictions.npz", uid=split["uid"][known], target=split["target"][known],
                            observed=y, neural=pred, item_baseline=baseline)
    write_json(out / "reward_metrics.json", report)
    return model


def fit_base(cfg, seed, train, val, rm, catalog, out, device):
    seed_all(seed, cfg["threads"])
    policy = Policy(cfg, len(catalog["counts"])).to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=cfg["base_lr"], weight_decay=0.01)
    rng = np.random.default_rng(seed)
    log = out / "base_training.jsonl"
    best_ndcg, start = -1, time.time()
    path = out / "base.pt"
    if log.exists():
        log.unlink()
    for step in range(1, cfg["base_steps"] + 1):
        policy.train()
        idx = rng.integers(len(train["x"]), size=cfg["batch_size"])
        x = tensor_x(train, idx, device)
        y = torch.tensor(train["target"][idx] - 2, device=device)
        logits = policy(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1)
        # Fixed warmup and cosine decay, fully determined before test evaluation.
        warmup = min(1, step / 100)
        decay = .1 + .9 * .5 * (1 + np.cos(np.pi * step / cfg["base_steps"]))
        for group in optimizer.param_groups:
            group["lr"] = cfg["base_lr"] * warmup * decay
        optimizer.step()
        if step % cfg["eval_every"] == 0 or step == cfg["base_steps"]:
            metrics, _, _ = evaluate(policy, rm, val, catalog, device)
            append_jsonl(log, {"step": step, "batch_loss": loss.item(), **metrics})
            if metrics["ndcg@10"] > best_ndcg:
                best_ndcg = metrics["ndcg@10"]
                save_weights(path, policy)
                write_json(out / "base_selection.json", {"step": step, "validation_ndcg@10": best_ndcg})
            print(f"seed={seed} base step={step} val NDCG@10={metrics['ndcg@10']:.5f} elapsed={time.time()-start:.1f}s", flush=True)
    load_weights(path, policy, device).eval()
    return policy, time.time() - start


def action_rewards(rm, x, actions, variant, novelty, duration):
    with torch.no_grad():
        completion = rm(x, actions)
        repeat = (actions[:, :, None] == x[:, None, :]).any(-1).float()
        if variant["reward"] == "completion":
            reward = completion
        elif variant["reward"] == "listening_time":
            # Expected fraction of a capped 10-minute window. Duration priors use
            # train events only, with replays deliberately capped in the target.
            reward = completion * duration[actions].clamp(max=600) / 600
        elif variant["reward"] == "balanced":
            reward = completion + variant["novelty_weight"] * novelty[actions] - variant["repeat_penalty"] * repeat
        else:
            raise ValueError(variant["reward"])
    return reward


def fit_rl(cfg, seed, base, train, val, rm, catalog, variant_name, out, device):
    # Every reward variant starts at exactly the same base checkpoint. Reset RNG
    # to make the contextual query stream and sampling seed comparable.
    seed_all(seed + 1000, cfg["threads"])
    policy = copy.deepcopy(base).to(device)
    reference = copy.deepcopy(base).to(device).eval()
    for p in reference.parameters():
        p.requires_grad_(False)
    variant = cfg["variants"][variant_name]
    optimizer = torch.optim.AdamW(policy.parameters(), lr=cfg["rl_lr"], weight_decay=0)
    rng = np.random.default_rng(seed + 1000)
    novelty = torch.tensor(catalog["novelty"], dtype=torch.float32, device=device)
    duration = torch.tensor(catalog["mean_duration_seconds"], dtype=torch.float32, device=device)
    log = out / f"{variant_name}_training.jsonl"
    if log.exists():
        log.unlink()
    start = time.time()
    for step in range(1, cfg["rl_steps"] + 1):
        # Disabling dropout makes the behavior distribution used for sampling
        # identical to the distribution differentiated by REINFORCE.
        policy.eval()
        idx = rng.integers(len(train["x"]), size=cfg["batch_size"])
        x = tensor_x(train, idx, device)
        target = torch.tensor(train["target"][idx] - 2, device=device)
        logp = policy(x).log_softmax(1)
        probs = logp.exp()
        with torch.no_grad():
            ref_logp = reference(x).log_softmax(1)
            actions_zero = torch.multinomial(probs.detach(), cfg["rl_samples_per_context"], replacement=True)
            rewards = action_rewards(rm, x, actions_zero + 2, variant, novelty, duration)
            # Leave-one-out baseline is independent of its own sampled action,
            # conditional on the context, and does not require a value network.
            baseline = (rewards.sum(1, keepdim=True) - rewards) / (rewards.shape[1] - 1)
            advantage = rewards - baseline
        reinforce_loss = -(advantage * logp.gather(1, actions_zero)).mean()
        kl = (probs * (logp - ref_logp)).sum(1).mean()
        supervised = torch.nn.functional.nll_loss(logp, target)
        loss = variant.get("rl_weight", 1.0) * reinforce_loss + variant["kl_beta"] * kl + cfg["rl_supervised_weight"] * supervised
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1)
        optimizer.step()
        if step % 100 == 0 or step == cfg["rl_steps"]:
            metrics, _, _ = evaluate(policy, rm, val, catalog, device)
            append_jsonl(log, {"step": step, "sampled_training_reward": rewards.mean().item(),
                              "kl_to_base": kl.item(), "supervised_nll": supervised.item(), **metrics})
            print(f"seed={seed} {variant_name} step={step} val NDCG@10={metrics['ndcg@10']:.5f} predicted completion={metrics['predicted_completion@10']:.4f}", flush=True)
    save_weights(out / f"{variant_name}.pt", policy)
    return policy.eval(), time.time() - start


def record_evaluation(name, policy, rm, val, test, catalog, cfg, out, device):
    entry = {}
    for split_name, split in [("validation", val), ("test", test)]:
        stats, per, top = evaluate(policy, rm, split, catalog, device)
        entry[split_name] = stats
        np.savez_compressed(out / f"{name}_{split_name}_per_query.npz", **per, top20=top, uid=split["uid"], target=split["target"])
    if not isinstance(policy, str):
        entry["rollout"] = rollout(policy, rm, test, catalog, cfg["rollout_horizon"], cfg["rollout_users"], cfg["rollout_discount"], device)
    write_json(out / f"{name}_evaluation.json", entry)
    return entry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/main.json")
    parser.add_argument("--data", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--reuse-reward", action="store_true")
    args = parser.parse_args()
    cfg = read_json(args.config)
    args.out.mkdir(parents=True, exist_ok=True)
    for name in ["train", "validation", "test"]:
        if not (args.data / f"{name}.npz").exists():
            parser.error("Run python -m recsys.data first")
    train, val, test = [load_split(args.data, s) for s in ["train", "validation", "test"]]
    catalog = read_json(args.data / "catalog.json")
    seed_all(17, cfg["threads"])
    env = {"python": platform.python_version(), "platform": platform.platform(), "device": args.device,
           "torch": torch.__version__, "threads": torch.get_num_threads(),
           "versions": {p: importlib.metadata.version(p) for p in ["numpy", "pyarrow", "transformers", "scikit-learn", "scipy"]},
           "config": cfg, "config_sha256": sha256(args.config),
           "data_files_sha256": {f"{s}.npz": sha256(args.data / f"{s}.npz") for s in ["train", "validation", "test"]}}
    write_json(args.out / "environment.json", env)
    if args.reuse_reward:
        rm = load_weights(args.out / "reward.pt", RewardModel(len(catalog["counts"]), cfg["reward_embedding_dim"]).to(args.device), args.device).eval()
        for p in rm.parameters():
            p.requires_grad_(False)
    else:
        rm = fit_reward(cfg, train, val, test, catalog, args.out, args.device)
    for name in ["popularity", "recency"]:
        record_evaluation(name, name, rm, val, test, catalog, cfg, args.out, args.device)
    for seed in args.seeds or cfg["seeds"]:
        out = args.out / f"seed_{seed}"
        out.mkdir(exist_ok=True)
        base, seconds = fit_base(cfg, seed, train, val, rm, catalog, out, args.device)
        summary = {"base": record_evaluation("base", base, rm, val, test, catalog, cfg, out, args.device)}
        summary["base"]["train_seconds"] = seconds
        summary["base"]["parameters"] = sum(p.numel() for p in base.parameters())
        write_json(out / "summary.json", summary)
        for name in cfg["variants"]:
            policy, seconds = fit_rl(cfg, seed, base, train, val, rm, catalog, name, out, args.device)
            summary[name] = record_evaluation(name, policy, rm, val, test, catalog, cfg, out, args.device)
            summary[name]["train_seconds"] = seconds
            write_json(out / "summary.json", summary)
    print("Experiment finished; raw metrics and checkpoints are in", args.out, flush=True)


if __name__ == "__main__":
    main()
