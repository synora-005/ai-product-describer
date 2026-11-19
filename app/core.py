"""Your code. One stub per milestone; build them in order and keep each one small."""

import re


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
def milestone_2():  # TODO(milestone 2)
    raise NotImplementedError("TODO(milestone 2)")


# --- Milestone 3: Build the causal self-attention mechanism and assemble the transformer block.
# Hint: Construct the causal self-attention layer and combine it with feed-forward networks within a transformer block.
def milestone_3():  # TODO(milestone 3)
    raise NotImplementedError("TODO(milestone 3)")


# --- Milestone 4: Train the GPT model using the collected product descriptions.
# Hint: Implement the training loop, including forward pass, loss calculation, and backpropagation with an optimizer.
def milestone_4():  # TODO(milestone 4)
    raise NotImplementedError("TODO(milestone 4)")


# --- Milestone 5: Implement a text generation feature that creates new product descriptions from a prompt.
# Hint: Develop a sampling strategy (e.g., greedy or top-k) to generate new tokens iteratively.
def milestone_5():  # TODO(milestone 5)
    raise NotImplementedError("TODO(milestone 5)")


# --- Milestone 6: Develop a CLI for users to input product details and receive generated descriptions.
# Hint: Structure your code to accept user input and display model output via the command line.
def milestone_6():  # TODO(milestone 6)
    raise NotImplementedError("TODO(milestone 6)")