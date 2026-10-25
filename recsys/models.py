"""Official Hugging Face GPT-2 backbone and an independent response regressor."""
import torch
from torch import nn
from transformers import GPT2Config, GPT2LMHeadModel


class Policy(nn.Module):
    def __init__(self, cfg, vocab):
        super().__init__()
        self.gpt = GPT2LMHeadModel(GPT2Config(
            vocab_size=vocab, n_positions=cfg["context_length"],
            n_ctx=cfg["context_length"], n_embd=cfg["embedding_dim"],
            n_layer=cfg["layers"], n_head=cfg["heads"],
            resid_pdrop=cfg["dropout"], embd_pdrop=cfg["dropout"],
            attn_pdrop=cfg["dropout"], pad_token_id=0,
            bos_token_id=None, eos_token_id=None, use_cache=False,
        ))

    def forward(self, x):
        mask = x.ne(0)
        # Project only the last observed position; avoids allocating B*T*V
        # logits when training on one explicitly sampled next-event target.
        hidden = self.gpt.transformer(input_ids=x, attention_mask=mask,
                                      use_cache=False).last_hidden_state
        index = mask.sum(1) - 1
        last = hidden[torch.arange(len(x), device=x.device), index]
        logits = self.gpt.lm_head(last)
        # PAD and OOV represent context, never recommendation actions.
        return logits[:, 2:]


class RewardModel(nn.Module):
    def __init__(self, vocab, dim=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab, dim, padding_idx=0)
        self.mlp = nn.Sequential(nn.Linear(4 * dim, 64), nn.ReLU(),
                                 nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

    def encode(self, x):
        mask = x.ne(0)
        embeddings = self.embedding(x)
        mean = (embeddings * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
        last = embeddings[torch.arange(len(x), device=x.device), mask.sum(1) - 1]
        return mean, last

    def predict_encoded(self, encoded, items):
        mean, last = encoded
        candidate = self.embedding(items)
        if items.ndim == 2:
            mean = mean[:, None, :].expand_as(candidate)
            last = last[:, None, :].expand_as(candidate)
        features = torch.cat([mean, last, candidate, mean * candidate], dim=-1)
        return self.mlp(features).squeeze(-1).sigmoid()

    def forward(self, x, items):
        return self.predict_encoded(self.encode(x), items)


def append_items(x, items):
    """Hypothetical rollout transition: append recommendation, not real feedback."""
    result = x.clone()
    lengths = x.ne(0).sum(1)
    full = lengths == x.shape[1]
    result[full, :-1] = x[full, 1:]
    positions = torch.clamp(lengths, max=x.shape[1] - 1)
    result[torch.arange(len(x), device=x.device), positions] = items
    return result
