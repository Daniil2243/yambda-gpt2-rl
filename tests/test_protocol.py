import numpy as np
import torch

from recsys.data import make_context
from recsys.evaluate import retrieval_per_query
from recsys.models import Policy, RewardModel, append_items


def small_config():
    return {"context_length": 5, "embedding_dim": 16, "layers": 1, "heads": 2, "dropout": 0.0}


def test_equal_timestamp_events_are_all_excluded():
    timestamps = np.array([1, 2, 2, 3])
    tokens = np.array([2, 3, 4, 5])
    x, end = make_context(tokens, timestamps, 2, 5)
    assert end == 1
    assert x.tolist() == [2, 0, 0, 0, 0]
    x2, _ = make_context(tokens, timestamps, 3, 2)
    assert x2.tolist() == [3, 4]


def test_oov_is_kept_in_context():
    x, _ = make_context(np.array([2, 1, 3, 4]), np.arange(4), 3, 5)
    assert x.tolist() == [2, 1, 3, 0, 0]


def test_retrieval_and_oov_target_miss():
    top = np.array([[3, 2, 4], [2, 3, 4], [4, 2, 3]])
    recall, ndcg = retrieval_per_query(top, np.array([2, 1, 4]), 2)
    np.testing.assert_allclose(recall, [1, 0, 1])
    np.testing.assert_allclose(ndcg, [1 / np.log2(3), 0, 1])


def test_append_padding_and_full_window():
    x = torch.tensor([[2, 3, 0], [2, 3, 4]])
    y = append_items(x, torch.tensor([5, 6]))
    assert y.tolist() == [[2, 3, 5], [3, 4, 6]]
    assert x.tolist() == [[2, 3, 0], [2, 3, 4]]


def test_policy_excludes_special_actions_and_ignores_future_positions():
    torch.manual_seed(1)
    model = Policy(small_config(), vocab=8).eval()
    x = torch.tensor([[2, 3, 4, 0, 0]])
    assert model(x).shape == (1, 6)
    # Directly check the GPT-2 causal mask: modifying later tokens must not
    # change the hidden states of earlier observed positions.
    a = model.gpt.transformer(torch.tensor([[2, 3, 4, 5, 6]])).last_hidden_state
    b = model.gpt.transformer(torch.tensor([[2, 3, 4, 7, 7]])).last_hidden_state
    torch.testing.assert_close(a[:, :3], b[:, :3])


def test_reward_is_history_conditioned_and_bounded():
    torch.manual_seed(3)
    rm = RewardModel(10, 4).eval()
    x = torch.tensor([[2, 3, 0], [4, 5, 0]])
    pred = rm(x, torch.tensor([[6, 7], [6, 7]]))
    assert pred.shape == (2, 2)
    assert ((pred > 0) & (pred < 1)).all()
    assert not torch.equal(pred[0], pred[1])


def test_exact_expected_reward_and_loo_gradient_point_toward_high_reward():
    # Enumerate two independent categorical samples to verify the estimator's
    # expected gradient, rather than only testing one random trajectory.
    logits = torch.tensor([0.2, -0.1], requires_grad=True)
    probs = logits.softmax(0)
    reward = torch.tensor([0.0, 1.0])
    true_loss = -(probs * reward).sum()
    true_grad = torch.autograd.grad(true_loss, logits, retain_graph=True)[0]
    expected_grad = torch.zeros(2)
    for a in range(2):
        for b in range(2):
            loss = -0.5 * ((reward[a] - reward[b]) * logits.log_softmax(0)[a]
                           + (reward[b] - reward[a]) * logits.log_softmax(0)[b])
            grad = torch.autograd.grad(loss, logits, retain_graph=True)[0]
            expected_grad += probs[a].detach() * probs[b].detach() * grad
    torch.testing.assert_close(expected_grad, true_grad)
