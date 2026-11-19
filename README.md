# AI Product Describer

PASTE THE BADGE LINE FROM THE ASKTHILI SHIP TAB HERE (replace this whole line)

A command-line tool that writes product descriptions. It is a small GPT (the same kind of model as ChatGPT, at a tiny scale) written from scratch in PyTorch and trained on T-shirt descriptions. You give it a few product details, and it continues them into a full description.

```
$ python -m app.cli describe v-neck t-shirt --variants 3
1. V-neck t-shirt in soft cotton jersey with short sleeves.
2. V-neck t-shirt in soft cotton jersey with a ribbed trim around the neckline and sleeves.
3. V-neck t-shirt in soft cotton jersey with a print motif on the front and short sleeves with sewn-in turn-ups.
```

Your exact output will differ, because it depends on how your own training run went.

## What you are building

**Goal.** A command-line tool that helps small business owners generate creative product descriptions by training a custom GPT on examples of their existing product texts.

**Done when.** A byte-pair tokenizer and embeddings prepare the text, causal self-attention and the transformer block are built, the GPT is trained from scratch, and it generates product descriptions through a working CLI.

## Milestones

- [x] **1. Collect a small corpus of product descriptions.** `make_corpus.py` builds it, and `load_corpus` cleans it (removes HTML and extra spaces, drops short lines and repeats).
- [x] **2. Byte-pair tokenizer and embeddings.** `BPETokenizer` learns merges from the corpus, and `Embeddings` adds a token vector and a position vector.
- [x] **3. Causal self-attention and the transformer block.** `CausalSelfAttention`, `FeedForward` and `TransformerBlock`.
- [x] **4. Train the GPT.** `GPT`, `train` and `estimate_loss`: forward pass, cross-entropy loss, backpropagation and the AdamW optimizer.
- [x] **5. Generate text from a prompt.** `generate` and `generate_description`, with greedy, temperature and top-k sampling.
- [x] **6. A CLI for product details.** `python -m app.cli` with `train`, `describe` and `interactive` commands.

## Set up

You need Python 3.12 or newer and git.

**Windows (PowerShell)**

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**Mac / Linux**

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Check that it works:

```
python -m app.cli hello
pytest
```

The starter's `.env.example` is optional. The GPT in this project does not read any settings from `.env`. If you do create a `.env` file, never commit it.

## Get the data

The training text is not in this repository. It comes from the product descriptions in the H&M Personalized Fashion Recommendations data on Kaggle, which is competition data. Check the competition's rules page before you use it, and do not redistribute it.

1. Create a free Kaggle account and open the [competition page](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations).
2. On the **Data** tab, click **Join the competition** and accept the rules.
3. Download **only `articles.csv`** (about 36 MB) with the download arrow on its card, not "Download All".
4. Save it as `data/articles.csv`.
5. Build the corpus:

```
python make_corpus.py
```

This keeps the T-shirt rows and writes `data/products.txt` with one unique description per line. For me it wrote 1,891 descriptions, about 200 KB.

You can also use your own text. Put one product description per line in `data/products.txt`, and use `--corpus` to point to a different file.

## Train the model

```
python -m app.cli train
```

This learns the tokenizer, trains the model for 1,000 steps and saves `data/tokenizer.json` and `data/model.pt`. It takes a few minutes on a laptop CPU, and it prints the training and validation loss every 100 steps. Options:

| Option | Default | What it does |
| --- | --- | --- |
| `--corpus FILE` | `data/products.txt` | text to learn from, one description per line |
| `--steps N` | 1000 | number of training steps |
| `--eval-every N` | 100 | how often to print the losses |
| `--vocab-size N` | 512 | tokenizer vocabulary size |
| `--model FILE` | `data/model.pt` | where to save the model |
| `--tokenizer FILE` | `data/tokenizer.json` | where to save the tokenizer |

## Use it

Write one description from product details:

```
python -m app.cli describe oversized cotton t-shirt
```

Write several versions and save them to a text file:

```
python -m app.cli describe striped polo shirt --variants 2 --save data/my_descriptions.txt
```

Keep typing product details (press Enter on an empty line to quit):

```
python -m app.cli interactive
```

Options for `describe` and `interactive`:

| Option | Default | What it does |
| --- | --- | --- |
| `--variants N` | 1 | how many different descriptions to write |
| `--temperature T` | 0.8 | 0 always picks the likeliest word, higher is more creative and messier |
| `--top-k K` | 40 | only pick from the K likeliest tokens (0 = no limit) |
| `--max-tokens N` | 60 | longest description, in tokens |
| `--seed N` | random | use the same number to repeat the same text |
| `--save FILE` | none | also append the descriptions to a text file (`describe` only) |
| `--model`, `--tokenizer` | `data/...` | which trained files to use |

A temperature around 0.5 to 0.8 gives believable text. At 1.2 the model starts inventing words.

## How it works

1. **Tokenize.** Byte-pair encoding starts from raw bytes and repeatedly merges the most common neighbouring pair into a new token, so common words like " jersey" become one token and rare words are spelled out from pieces. An end-of-description token marks where each description stops.
2. **Embed.** Each token ID becomes a learned vector, and a second learned vector says where in the text it is.
3. **Attend.** Causal self-attention lets each token look at the tokens before it and weigh how relevant each one is. A mask stops it from looking ahead. Several heads run in parallel.
4. **Stack blocks.** Each transformer block adds a small feed-forward network, with LayerNorm and residual connections to keep training stable. The model stacks four of them, then turns each position into a score for every token in the vocabulary.
5. **Train.** Take random 64-token windows, predict the next token at every position, measure the cross-entropy loss, backpropagate and update the weights with AdamW. The last 10% of the text is held out as a validation set.
6. **Generate.** Start from the end token plus the product details, score the next token, pick one (greedy, or sampled with temperature and top-k), append it and repeat until the end token appears.

**Model size:** vocabulary 512, 64 tokens of context, 64-dimensional vectors, 4 heads, 4 layers, about 270,000 parameters. In my run the validation loss reached about 1.8 (roughly "choosing between 6 likely tokens"), while the training loss kept falling, so the model starts to memorise the corpus after a few hundred steps.

## Tests

```
pytest
```

The tests use tiny made-up text and untrained or briefly trained small models, so they run without the dataset. They check the cleaning, tokenizer round trips, that attention never looks ahead, that training lowers the loss on an easy pattern, the sampling rules and the CLI commands.

## Limits

- The model is tiny and learned from about 56,000 tokens, so it writes in one catalogue style and sometimes produces glitches, such as missing spaces or invented words.
- **It makes up details.** It can say a product has a pocket or a certain fabric when it does not. Always check a generated description against the real product before using it.
- It only knows T-shirts and similar tops. Other product types will come out as T-shirt-like text.
- The data is not mine to redistribute, so it is not in this repo.

## Ideas to try next

- Add a tone option (formal, playful) by putting a tone word at the start of every training description.
- Stop training at the step where the validation loss is lowest and keep that version.
- Train on a bigger, more varied corpus, or on a different category such as shoes.
- Penalise repeated phrases while generating.