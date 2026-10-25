"""Full-catalog retrieval, frozen-reward diagnostics, and hypothetical rollouts."""
import numpy as np
import torch

from .models import append_items


def retrieval_per_query(top, targets, k):
    hit = top[:, :k] == targets[:, None]
    recall = hit.any(axis=1).astype(np.float64)
    ndcg = (hit / np.log2(np.arange(2, k + 2))).sum(axis=1)
    return recall, ndcg


def tensor_x(split, indices, device):
    return torch.as_tensor(split["x"][indices], dtype=torch.long, device=device)


@torch.inference_mode()
def evaluate(policy, reward_model, split, catalog, device="cpu", batch_size=256):
    if policy is not None and not isinstance(policy, str):
        policy.eval()
    if reward_model is not None:
        reward_model.eval()
    popularity = np.asarray(catalog["counts"])
    rank = np.argsort(-popularity[2:], kind="stable") + 2
    # Last-item recency, then unique recent items, then train popularity.
    all_top, scores, entropy = [], [], []
    predicted, sampled_reward, repeated, novelty_values = [], [], [], []
    duration_values, sample_repeat = [], []
    gen = torch.Generator(device=device).manual_seed(20261025)
    novel = torch.tensor(catalog["novelty"], device=device)
    durations = torch.tensor(catalog["mean_duration_seconds"], device=device)
    for start in range(0, len(split["x"]), batch_size):
        indices = np.arange(start, min(start + batch_size, len(split["x"])))
        x = tensor_x(split, indices, device)
        if isinstance(policy, str):
            rows = []
            for row in x.cpu().numpy():
                candidates = list(dict.fromkeys(int(v) for v in row[::-1] if v >= 2)) if policy == "recency" else []
                candidates += [int(v) for v in rank if v not in candidates][:20]
                rows.append(candidates[:20])
            top = torch.tensor(rows, device=device)
        else:
            logits = policy(x)
            logp = logits.log_softmax(1)
            probs = logp.exp()
            top = logits.topk(20, dim=1).indices + 2
            scores.append(logp.gather(1, torch.as_tensor(np.maximum(split["target"][indices] - 2, 0), device=device)[:, None]).squeeze(1).cpu().numpy())
            entropy.append((-(probs * logp).sum(1)).cpu().numpy())
            action = torch.multinomial(probs, 16, replacement=True, generator=gen) + 2
            sampled_reward.append(reward_model(x, action).mean(1).cpu().numpy())
            sample_repeat.append((action[:, :, None] == x[:, None, :]).any(2).float().mean(1).cpu().numpy())
        all_top.append(top.cpu().numpy())
        if reward_model is not None:
            predicted.append(reward_model(x, top[:, :10]).cpu().numpy())
        repeated.append((top[:, :10, None] == x[:, None, :]).any(2).float().mean(1).cpu().numpy())
        novelty_values.append(novel[top[:, :10]].mean(1).cpu().numpy())
        duration_values.append(durations[top[:, :10]].mean(1).cpu().numpy())
    top = np.concatenate(all_top)
    known = split["target"] >= 2
    per = {}
    for k in [5, 10, 20]:
        recall, ndcg = retrieval_per_query(top, split["target"], k)
        per[f"recall@{k}"] = recall
        per[f"ndcg@{k}"] = ndcg
    stats = {name: float(values.mean()) for name, values in per.items()}
    stats.update({f"catalog_{name}": float(values[known].mean()) for name, values in per.items()})
    repeat = np.concatenate(repeated)
    novelty = np.concatenate(novelty_values)
    duration = np.concatenate(duration_values)
    stats.update({"queries": len(top), "known_targets": int(known.sum()),
                  "query_catalog_coverage": float(known.mean()), "history_repeat@10": float(repeat.mean()),
                  "novelty@10": float(novelty.mean()), "mean_duration@10_seconds": float(duration.mean()),
                  "catalog_coverage@10": len(np.unique(top[:, :10])) / (len(catalog["counts"]) - 2),
                  "top1_max_share": float(np.bincount(top[:, 0]).max() / len(top))})
    per.update({"history_repeat@10": repeat, "novelty@10": novelty})
    if predicted:
        predicted = np.concatenate(predicted)
        per["predicted_completion@10"] = predicted.mean(1)
        per["predicted_completion@1"] = predicted[:, 0]
        stats["predicted_completion@10"] = float(predicted.mean())
        stats["predicted_completion@1"] = float(predicted[:, 0].mean())
    if entropy:
        stats["policy_entropy"] = float(np.concatenate(entropy).mean())
        stats["catalog_nll"] = float(-np.concatenate(scores)[known].mean())
        per["sampled_completion"] = np.concatenate(sampled_reward)
        stats["sampled_completion"] = float(per["sampled_completion"].mean())
        stats["sampled_history_repeat"] = float(np.concatenate(sample_repeat).mean())
    return stats, per, top


@torch.inference_mode()
def rollout(policy, reward_model, split, catalog, horizon=5, users=512, gamma=0.9, device="cpu"):
    """Greedy generated suffixes. No claim of real session or causal user utility."""
    policy.eval()
    reward_model.eval()
    x = tensor_x(split, np.arange(min(users, len(split["x"]))), device)
    original = x.clone()
    actions, rewards, fresh_rewards = [], [], []
    for step in range(horizon):
        item = policy(x).argmax(1) + 2
        rewards.append(reward_model(x, item).cpu().numpy())
        fresh_rewards.append(reward_model(original, item).cpu().numpy())
        actions.append(item.cpu().numpy())
        x = append_items(x, item)
    actions = np.stack(actions, 1)
    rewards = np.stack(rewards, 1)
    original_np = original.cpu().numpy()
    duplicates = np.array([1 - len(set(row)) / horizon for row in actions])
    history_repeat = (actions[:, :, None] == original_np[:, None, :]).any(2).mean(1)
    return {"users": len(x), "horizon": horizon, "discount": gamma,
            "discounted_model_return": float((rewards * gamma**np.arange(horizon)).sum(1).mean()),
            "mean_predicted_completion_by_step": rewards.mean(0).tolist(),
            "original_context_completion_by_step": np.stack(fresh_rewards, 1).mean(0).tolist(),
            "within_slate_duplicate_fraction": float(duplicates.mean()),
            "original_history_repeat_fraction": float(history_repeat.mean()),
            "unique_generated_items": int(len(np.unique(actions))),
            "examples": [{"uid": int(split["uid"][i]), "history_tokens": original_np[i][original_np[i] > 0].tolist(),
                          "generated_item_ids": [catalog["item_ids"][int(v)] for v in actions[i]],
                          "predicted_rewards": rewards[i].tolist()} for i in range(min(5, len(x)))]}
