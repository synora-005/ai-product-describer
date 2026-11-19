"""Your code. One stub per milestone; build them in order and keep each one small."""

import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


# --- Milestone 1: Collect a small corpus of product descriptions for a specific category.
# Hint: Focus on gathering raw text data and basic cleaning for your corpus.
def clean_description(text):
    """Remove HTML tags and extra spaces from one description."""
    text = re.sub(r"<[^>]+>", " ", text)   # drop things like <b> and </b>
    text = re.sub(r"\s+", " ", text)       # many spaces or tabs become one space
    return text.strip()


def load_corpus(path, min_chars=30):
    """Read a file with one description per line.

    Cleans each line, drops lines that are too short, and removes repeats.
    Returns a list of descriptions in their original order.
    """
    descriptions = []
    seen = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            text = clean_description(line)
            if len(text) < min_chars or text in seen:
                continue
            seen.add(text)
            descriptions.append(text)
    return descriptions


def corpus_stats(descriptions):
    """Return a few numbers that describe the corpus."""
    count = len(descriptions)
    characters = sum(len(d) for d in descriptions)
    words = sum(len(d.split()) for d in descriptions)
    return {
        "count": count,
        "characters": characters,
        "avg_words": round(words / count, 1) if count else 0.0,
    }


# --- Milestone 2: Implement the byte-pair tokenizer and generate embeddings for product text.
# Hint: Apply byte-pair encoding for tokenization and create an embedding layer for numerical representation.

# Text is first cut into chunks (a word with its leading space, a run of punctuation,
# or whitespace). Merges never cross a chunk boundary.
CHUNKS = re.compile(r" ?\w+| ?[^\w\s]+|\s+")


def merge_pair(ids, pair, new_id):
    """Replace every neighbouring (a, b) in ids with new_id."""
    out = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    """Byte-level byte-pair encoding.

    IDs 0-255 are single bytes, 256 and up are merged pairs (in the order they were
    learned), and the last ID is the end-of-text token.
    """

    def __init__(self, merges=()):
        self.merges = [tuple(pair) for pair in merges]
        self.ranks = {pair: 256 + i for i, pair in enumerate(self.merges)}
        self.vocab = {i: bytes([i]) for i in range(256)}
        for i, (a, b) in enumerate(self.merges):
            self.vocab[256 + i] = self.vocab[a] + self.vocab[b]
        self.eot_id = 256 + len(self.merges)
        self.vocab_size = self.eot_id + 1
        self._cache = {}

    @classmethod
    def train(cls, texts, vocab_size):
        """Learn merges from a list of texts. vocab_size counts bytes, merges and the end token."""
        num_merges = max(0, vocab_size - 257)

        # count each distinct chunk once, with how often it occurs (much faster)
        chunk_counts = Counter()
        for text in texts:
            for chunk in CHUNKS.findall(text):
                chunk_counts[tuple(chunk.encode("utf-8"))] += 1
        words = [(list(chunk), count) for chunk, count in chunk_counts.items()]

        merges = []
        for _ in range(num_merges):
            pair_counts = Counter()
            for ids, count in words:
                for pair in zip(ids, ids[1:]):
                    pair_counts[pair] += count
            if not pair_counts:
                break
            best = max(pair_counts, key=lambda p: (pair_counts[p], p))
            new_id = 256 + len(merges)
            merges.append(best)
            words = [(merge_pair(ids, best, new_id), count) for ids, count in words]
        return cls(merges)

    def _encode_chunk(self, chunk):
        if chunk in self._cache:
            return self._cache[chunk]
        ids = list(chunk.encode("utf-8"))
        while len(ids) >= 2:
            pairs = set(zip(ids, ids[1:]))
            best = min(pairs, key=lambda p: self.ranks.get(p, float("inf")))
            if best not in self.ranks:
                break
            ids = merge_pair(ids, best, self.ranks[best])
        self._cache[chunk] = ids
        return ids

    def encode(self, text):
        """Turn text into a list of token IDs."""
        ids = []
        for chunk in CHUNKS.findall(text):
            ids.extend(self._encode_chunk(chunk))
        return ids

    def decode(self, ids):
        """Turn token IDs back into text. The end-of-text token becomes nothing."""
        data = b"".join(self.vocab.get(i, b"") for i in ids)
        return data.decode("utf-8", errors="replace")

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"merges": self.merges}, f)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f)["merges"])


def encode_corpus(descriptions, tokenizer):
    """Encode every description and put the end-of-text token after each one.
    Returns one long list of token IDs."""
    ids = []
    for description in descriptions:
        ids.extend(tokenizer.encode(description))
        ids.append(tokenizer.eot_id)
    return ids


class Embeddings(nn.Module):
    """Token embedding plus position embedding.

    Input: token IDs of shape (batch, length). Output: vectors of shape (batch, length, n_embd).
    """

    def __init__(self, vocab_size, block_size, n_embd):
        super().__init__()
        self.block_size = block_size
        self.token = nn.Embedding(vocab_size, n_embd)
        self.position = nn.Embedding(block_size, n_embd)

    def forward(self, idx):
        _, length = idx.shape
        if length > self.block_size:
            raise ValueError(f"sequence length {length} is longer than block_size {self.block_size}")
        positions = torch.arange(length, device=idx.device)
        return self.token(idx) + self.position(positions)


# --- Milestone 3: Build the causal self-attention mechanism and assemble the transformer block.
# Hint: Construct the causal self-attention layer and combine it with feed-forward networks within a transformer block.
class CausalSelfAttention(nn.Module):
    """Several attention heads in parallel. Each token can only look at itself and earlier tokens.

    Input and output both have shape (batch, length, n_embd).
    """

    def __init__(self, n_embd, n_head, block_size, dropout=0.0):
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError(f"n_embd ({n_embd}) must be divisible by n_head ({n_head})")
        self.n_head = n_head
        self.block_size = block_size
        self.qkv = nn.Linear(n_embd, 3 * n_embd)   # makes queries, keys and values in one go
        self.proj = nn.Linear(n_embd, n_embd)      # mixes the heads back together
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        # lower-triangular matrix of ones: row t may look at columns 0..t only
        mask = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer("mask", mask.view(1, 1, block_size, block_size))

    def forward(self, x):
        batch, length, channels = x.shape
        if length > self.block_size:
            raise ValueError(f"sequence length {length} is longer than block_size {self.block_size}")
        head_size = channels // self.n_head

        q, k, v = self.qkv(x).split(channels, dim=2)
        # (batch, length, channels) -> (batch, heads, length, head_size)
        q = q.view(batch, length, self.n_head, head_size).transpose(1, 2)
        k = k.view(batch, length, self.n_head, head_size).transpose(1, 2)
        v = v.view(batch, length, self.n_head, head_size).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(head_size)         # (batch, heads, length, length)
        scores = scores.masked_fill(self.mask[:, :, :length, :length] == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)                               # each row adds up to 1
        weights = self.attn_dropout(weights)

        out = weights @ v                                                 # (batch, heads, length, head_size)
        out = out.transpose(1, 2).contiguous().view(batch, length, channels)
        return self.resid_dropout(self.proj(out))


class FeedForward(nn.Module):
    """A small two-layer network applied to every position separately."""

    def __init__(self, n_embd, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """Attention, then feed-forward, each with LayerNorm in front and a residual (x + ...) around it."""

    def __init__(self, n_embd, n_head, block_size, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffn = FeedForward(n_embd, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


# --- Milestone 4: Train the GPT model using the collected product descriptions.
# Hint: Implement the training loop, including forward pass, loss calculation, and backpropagation with an optimizer.
@dataclass
class GPTConfig:
    """The size settings of the model."""
    vocab_size: int
    block_size: int = 64    # how many tokens the model can look at
    n_embd: int = 64        # size of each token's vector
    n_head: int = 4         # attention heads per block
    n_layer: int = 4        # number of transformer blocks
    dropout: float = 0.1    # randomly switch off 10% of values while training (fights memorising)


class GPT(nn.Module):
    """Embeddings -> transformer blocks -> LayerNorm -> one score per token in the vocabulary."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = Embeddings(config.vocab_size, config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.Sequential(*[
            TransformerBlock(config.n_embd, config.n_head, config.block_size, config.dropout)
            for _ in range(config.n_layer)
        ])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        # small random starting values, so the first guesses are close to "anything is equally likely"
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None):
        """idx: token IDs (batch, length). Returns scores (batch, length, vocab_size) and,
        if targets are given, the loss (how badly the next tokens were predicted)."""
        x = self.drop(self.embeddings(idx))
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


def train_val_split(ids, val_fraction=0.1):
    """Keep the last part of the data aside to check the model on text it has not trained on."""
    data = torch.tensor(ids, dtype=torch.long)
    cut = int(len(data) * (1 - val_fraction))
    return data[:cut], data[cut:]


def get_batch(data, block_size, batch_size):
    """Pick random windows. y is x shifted one token to the left: the 'next token' at every position."""
    starts = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in starts])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in starts])
    return x, y


def estimate_loss(model, data, block_size, batch_size=32, eval_batches=20):
    """Average loss over a few random batches, without training."""
    was_training = model.training
    model.eval()
    total = 0.0
    with torch.no_grad():
        for _ in range(eval_batches):
            x, y = get_batch(data, block_size, batch_size)
            _, loss = model(x, y)
            total += loss.item()
    model.train(was_training)
    return total / eval_batches


def train(model, train_data, val_data=None, *, block_size, steps=1000, batch_size=32,
          lr=3e-3, eval_every=0, eval_batches=20, log=None):
    """The training loop. Returns a list of {"step", "train_loss", "val_loss"} records."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    history = []
    model.train()
    for step in range(1, steps + 1):
        x, y = get_batch(train_data, block_size, batch_size)
        _, loss = model(x, y)                                   # forward pass and loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()                                         # backpropagation
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # stop one huge step from wrecking things
        optimizer.step()                                        # nudge every number to lower the loss

        if eval_every and (step % eval_every == 0 or step == steps):
            record = {"step": step,
                      "train_loss": estimate_loss(model, train_data, block_size, batch_size, eval_batches)}
            if val_data is not None:
                record["val_loss"] = estimate_loss(model, val_data, block_size, batch_size, eval_batches)
            history.append(record)
            if log:
                log(record)
    return history


def save_model(model, path):
    """Save the settings and the learned numbers in one file."""
    torch.save({"config": asdict(model.config), "state": model.state_dict()}, path)


def load_model(path):
    """Load a model saved with save_model, ready for use (not for training)."""
    checkpoint = torch.load(path, map_location="cpu")
    model = GPT(GPTConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["state"])
    model.eval()
    return model


# --- Milestone 5: Implement a text generation feature that creates new product descriptions from a prompt.
# Hint: Develop a sampling strategy (e.g., greedy or top-k) to generate new tokens iteratively.
def milestone_5():  # TODO(milestone 5)
    raise NotImplementedError("TODO(milestone 5)")


# --- Milestone 6: Develop a CLI for users to input product details and receive generated descriptions.
# Hint: Structure your code to accept user input and display model output via the command line.
def milestone_6():  # TODO(milestone 6)
    raise NotImplementedError("TODO(milestone 6)")