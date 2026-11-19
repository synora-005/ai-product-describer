"""Your tests. Each milestone in README.md gets one here: write the test first, then make it pass.

They start skipped. Remove the `skip` line as you work on each milestone.
"""
import math

import pytest
import torch

from app.core import (
    BPETokenizer,
    CausalSelfAttention,
    Embeddings,
    GPT,
    GPTConfig,
    TransformerBlock,
    corpus_stats,
    encode_corpus,
    estimate_loss,
    generate,
    generate_description,
    get_batch,
    load_corpus,
    load_model,
    save_model,
    train,
    train_val_split,
)


# Milestone 1: Collect a small corpus of product descriptions for a specific category.
def test_milestone_1(tmp_path):
    path = tmp_path / "products.txt"
    path.write_text(
        "Top in soft  jersey with <b>short</b> sleeves.\n"  # messy spacing and HTML
        "\n"                                                 # empty line
        "Top in soft jersey with short sleeves.\n"           # same as the first once cleaned
        "Tiny\n"                                             # too short
        "Fitted top with a V-neck.\n",
        encoding="utf-8",
    )
    descriptions = load_corpus(str(path), min_chars=10)
    assert descriptions == [
        "Top in soft jersey with short sleeves.",
        "Fitted top with a V-neck.",
    ]

    stats = corpus_stats(["a b c", "d e"])
    assert stats == {"count": 2, "characters": 8, "avg_words": 2.5}


# Milestone 2: Implement the byte-pair tokenizer and generate embeddings for product text.
def test_milestone_2():
    texts = ["the cat the cat the cat the cat"]
    tokenizer = BPETokenizer.train(texts, vocab_size=257 + 10)

    # encoding then decoding gives the same text back, even for characters it never saw
    for text in ["the cat", "the cat café ☕", "  extra   spaces!  "]:
        assert tokenizer.decode(tokenizer.encode(text)) == text

    # learned merges make the token list shorter than the raw bytes
    assert len(tokenizer.encode("the cat")) < len("the cat".encode("utf-8"))

    # the end-of-text token comes after all bytes and merges, and decodes to nothing
    assert tokenizer.eot_id == 256 + len(tokenizer.merges)
    assert tokenizer.vocab_size == tokenizer.eot_id + 1
    assert tokenizer.decode([tokenizer.eot_id]) == ""

    # training twice on the same text learns the same merges
    assert BPETokenizer.train(texts, vocab_size=257 + 10).merges == tokenizer.merges

    # the corpus becomes one long list of IDs, with the end token after each description
    ids = encode_corpus(["the cat", "the cat"], tokenizer)
    assert ids.count(tokenizer.eot_id) == 2
    assert ids[-1] == tokenizer.eot_id


def test_milestone_2_embeddings():
    embeddings = Embeddings(vocab_size=50, block_size=8, n_embd=16)
    tokens = torch.randint(0, 50, (2, 8))
    assert embeddings(tokens).shape == (2, 8, 16)

    # a sequence longer than block_size is refused
    with pytest.raises(ValueError):
        embeddings(torch.randint(0, 50, (1, 9)))


# Milestone 3: Build the causal self-attention mechanism and assemble the transformer block.
def test_milestone_3():
    torch.manual_seed(0)
    attention = CausalSelfAttention(n_embd=16, n_head=4, block_size=8)
    x = torch.randn(2, 8, 16)
    assert attention(x).shape == (2, 8, 16)

    # causal: changing the last three tokens must not change the outputs for the first five
    changed = x.clone()
    changed[:, 5:, :] = torch.randn(2, 3, 16)
    before, after = attention(x), attention(changed)
    assert torch.allclose(before[:, :5], after[:, :5], atol=1e-6)
    assert not torch.allclose(before[:, 5:], after[:, 5:])

    # the number of heads has to divide the embedding size
    with pytest.raises(ValueError):
        CausalSelfAttention(n_embd=16, n_head=3, block_size=8)

    # a sequence longer than block_size is refused
    with pytest.raises(ValueError):
        attention(torch.randn(1, 9, 16))


def test_milestone_3_block():
    torch.manual_seed(0)
    block = TransformerBlock(n_embd=16, n_head=4, block_size=8)
    x = torch.randn(2, 8, 16)
    out = block(x)
    assert out.shape == (2, 8, 16)

    # the block is causal too: the first 5 outputs do not depend on later tokens
    assert torch.allclose(block(x[:, :5]), out[:, :5], atol=1e-5)

    # training can reach every part of the block
    out.sum().backward()
    assert all(p.grad is not None for p in block.parameters())


# Milestone 4: Train the GPT model using the collected product descriptions.
def small_config(vocab_size):
    return GPTConfig(vocab_size=vocab_size, block_size=8, n_embd=16, n_head=2, n_layer=2, dropout=0.0)


def test_milestone_4():
    torch.manual_seed(0)

    # batches: y is x moved along by one token
    data = torch.arange(100)
    x, y = get_batch(data, block_size=8, batch_size=4)
    assert x.shape == (4, 8) and y.shape == (4, 8)
    assert torch.equal(y, x + 1)

    # the split keeps the last 10% for checking
    train_data, val_data = train_val_split(list(range(100)), val_fraction=0.1)
    assert len(train_data) == 90 and len(val_data) == 10

    # before training, the model is guessing: loss is close to ln(vocab size)
    model = GPT(small_config(vocab_size=50))
    tokens = torch.randint(0, 50, (4, 8))
    logits, loss = model(tokens, tokens)
    assert logits.shape == (4, 8, 50)
    assert abs(loss.item() - math.log(50)) < 0.3


def test_milestone_4_training():
    torch.manual_seed(0)
    data = torch.tensor([0, 1, 2, 3] * 50)   # a pattern that is easy to learn: 0 1 2 3 0 1 2 3 ...
    model = GPT(small_config(vocab_size=4))

    before = estimate_loss(model, data, block_size=8, batch_size=8, eval_batches=5)
    history = train(model, data, data, block_size=8, steps=200, batch_size=8, lr=1e-2, eval_every=100)
    after = estimate_loss(model, data, block_size=8, batch_size=8, eval_batches=5)

    assert after < before * 0.5
    assert [record["step"] for record in history] == [100, 200]
    assert "val_loss" in history[0]


def test_milestone_4_save_and_load(tmp_path):
    torch.manual_seed(0)
    model = GPT(small_config(vocab_size=20))
    model.eval()
    path = str(tmp_path / "model.pt")
    save_model(model, path)

    loaded = load_model(path)
    tokens = torch.randint(0, 20, (2, 8))
    assert torch.allclose(model(tokens)[0], loaded(tokens)[0])


# Milestone 5: Implement a text generation feature that creates new product descriptions from a prompt.
def test_milestone_5():
    torch.manual_seed(0)
    model = GPT(small_config(vocab_size=50))
    prompt = [1, 2, 3]

    # greedy (temperature 0) always gives the same answer, and respects the length limit
    greedy = generate(model, prompt, max_new_tokens=12, temperature=0, top_k=None)
    assert greedy == generate(model, prompt, max_new_tokens=12, temperature=0, top_k=None)
    assert len(greedy) == 12

    # top_k=1 leaves only the best token, so it matches greedy
    assert generate(model, prompt, max_new_tokens=12, temperature=1.0, top_k=1) == greedy

    # the same seed repeats the same text; different seeds give different text
    first = generate(model, prompt, max_new_tokens=20, temperature=1.0, top_k=None, seed=1)
    assert first == generate(model, prompt, max_new_tokens=20, temperature=1.0, top_k=None, seed=1)
    assert first != generate(model, prompt, max_new_tokens=20, temperature=1.0, top_k=None, seed=2)

    # bad settings are refused
    with pytest.raises(ValueError):
        generate(model, prompt, temperature=-1)
    with pytest.raises(ValueError):
        generate(model, prompt, top_k=0)


def test_milestone_5_stops_at_end_token():
    torch.manual_seed(0)
    # the pattern 0 1 2 <end> repeated, where the end token is ID 4
    data = torch.tensor([0, 1, 2, 4] * 60)
    model = GPT(small_config(vocab_size=5))
    train(model, data, block_size=8, steps=300, batch_size=8, lr=1e-2)

    # after "0" comes 1, then 2, then the end token, so generation stops with [1, 2]
    assert generate(model, [4, 0], max_new_tokens=10, temperature=0, top_k=None, eot_id=4) == [1, 2]

    # without an end token to stop at, it keeps going until the limit
    assert generate(model, [4, 0], max_new_tokens=5, temperature=0, top_k=None) == [1, 2, 4, 0, 1]


def test_milestone_5_description():
    torch.manual_seed(0)
    tokenizer = BPETokenizer.train(["the cat sat on the mat"], vocab_size=257 + 5)
    model = GPT(small_config(vocab_size=tokenizer.vocab_size))

    text = generate_description(model, tokenizer, "the cat", max_new_tokens=10, seed=0)
    assert isinstance(text, str)
    assert text.startswith("the cat")
    assert text == generate_description(model, tokenizer, "the cat", max_new_tokens=10, seed=0)


# Milestone 6: Develop a CLI for users to input product details and receive generated descriptions.
@pytest.mark.skip(reason="remove this line when you start milestone 6")
def test_milestone_6():
    raise NotImplementedError("write the test for milestone 6")