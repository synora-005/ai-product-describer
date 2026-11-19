# AI Product Describer

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/ndwvqx2mxn/badge.svg)](https://ask.thili.ai/b/ndwvqx2mxn)

This project delivers a practical CLI tool for small business owners, enabling them to effortlessly generate unique product descriptions. By training a custom GPT model on existing product texts, the tool learns to craft creative and relevant descriptions, streamlining the inventory management process.

## What you are building

**Goal.** You will build a command-line interface (CLI) tool that helps small business owners generate creative product descriptions by training a custom GPT model on examples of their existing product texts.

**Done when.** You will have successfully applied byte-pair tokenizer and embeddings to prepare raw text, implemented causal self-attention, assembled the transformer block, and trained the GPT model from scratch to generate coherent product descriptions via a functional CLI.

## Milestones

Tick these off as you go, and commit after each one.

- [ ] **1. Collect a small corpus of product descriptions for a specific category.**  
      Hint: Focus on gathering raw text data and basic cleaning for your corpus.
- [ ] **2. Implement the byte-pair tokenizer and generate embeddings for product text.**  
      Hint: Apply byte-pair encoding for tokenization and create an embedding layer for numerical representation.
- [ ] **3. Build the causal self-attention mechanism and assemble the transformer block.**  
      Hint: Construct the causal self-attention layer and combine it with feed-forward networks within a transformer block.
- [ ] **4. Train the GPT model using the collected product descriptions.**  
      Hint: Implement the training loop, including forward pass, loss calculation, and backpropagation with an optimizer.
- [ ] **5. Implement a text generation feature that creates new product descriptions from a prompt.**  
      Hint: Develop a sampling strategy (e.g., greedy or top-k) to generate new tokens iteratively.
- [ ] **6. Develop a CLI for users to input product details and receive generated descriptions.**  
      Hint: Structure your code to accept user input and display model output via the command line.

## Run it

You need Python 3.12 and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then choose a model; see the comments inside
python -m app.cli hello
pytest
```

Your work goes in `app/core.py` (one stub per milestone) and `tests/test_core.py`. The first tests pass already; they
check the plumbing.

## Settings and secrets

Model settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.** If a key ever
reaches GitHub, treat it as leaked and make a new one.

## Stretch ideas

- Allow the user to specify a 'tone' for the description (e.g., formal, playful) by adding a prompt prefix.
- Implement a feature to generate multiple description variants for the same product details.
- Add an option to save generated descriptions directly to a text file or copy to clipboard.
