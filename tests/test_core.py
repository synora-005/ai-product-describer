"""Your tests. Each milestone in README.md gets one here: write the test first, then make it pass.

They start skipped. Remove the `skip` line as you work on each milestone.
"""
import pytest
import torch

from app.core import BPETokenizer, Embeddings, corpus_stats, encode_corpus, load_corpus


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
@pytest.mark.skip(reason="remove this line when you start milestone 3")
def test_milestone_3():
    raise NotImplementedError("write the test for milestone 3")


# Milestone 4: Train the GPT model using the collected product descriptions.
@pytest.mark.skip(reason="remove this line when you start milestone 4")
def test_milestone_4():
    raise NotImplementedError("write the test for milestone 4")


# Milestone 5: Implement a text generation feature that creates new product descriptions from a prompt.
@pytest.mark.skip(reason="remove this line when you start milestone 5")
def test_milestone_5():
    raise NotImplementedError("write the test for milestone 5")


# Milestone 6: Develop a CLI for users to input product details and receive generated descriptions.
@pytest.mark.skip(reason="remove this line when you start milestone 6")
def test_milestone_6():
    raise NotImplementedError("write the test for milestone 6")