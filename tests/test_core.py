"""Your tests. Each milestone in README.md gets one here: write the test first, then make it pass.

They start skipped. Remove the `skip` line as you work on each milestone.
"""
import pytest

from app.core import load_corpus, corpus_stats


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
@pytest.mark.skip(reason="remove this line when you start milestone 2")
def test_milestone_2():
    raise NotImplementedError("write the test for milestone 2")


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