import torch

from app.core import (
    BPETokenizer, GPT, GPTConfig, encode_corpus, load_corpus, save_model, train, train_val_split,
)

STEPS = 1000        # how many training steps; lower it if it takes too long
EVAL_EVERY = 100    # print the losses this often

torch.manual_seed(0)

descriptions = load_corpus("data/products.txt")
tokenizer = BPETokenizer.load("data/tokenizer.json")
ids = encode_corpus(descriptions, tokenizer)
train_data, val_data = train_val_split(ids)
print("training tokens:", len(train_data), " validation tokens:", len(val_data))

config = GPTConfig(vocab_size=tokenizer.vocab_size)
model = GPT(config)
print("model parameters:", sum(p.numel() for p in model.parameters()))


def show(record):
    print(f"step {record['step']:>5}   train loss {record['train_loss']:.3f}   val loss {record['val_loss']:.3f}")


train(model, train_data, val_data, block_size=config.block_size, steps=STEPS,
      eval_every=EVAL_EVERY, log=show)

save_model(model, "data/model.pt")
print("saved data/model.pt")