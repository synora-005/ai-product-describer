import torch

from app.core import BPETokenizer, Embeddings, encode_corpus, load_corpus

descriptions = load_corpus("data/products.txt")

tokenizer = BPETokenizer.train(descriptions, vocab_size=512)
tokenizer.save("data/tokenizer.json")
print("vocab size:", tokenizer.vocab_size)

sample = descriptions[0]
ids = tokenizer.encode(sample)
print(sample)
print(len(sample.encode("utf-8")), "bytes ->", len(ids), "tokens")
print([tokenizer.decode([i]) for i in ids])
print("round trip ok:", tokenizer.decode(ids) == sample)

all_ids = encode_corpus(descriptions, tokenizer)
print("corpus tokens:", len(all_ids))

embeddings = Embeddings(tokenizer.vocab_size, block_size=64, n_embd=32)
x = torch.tensor(all_ids[:64]).unsqueeze(0)
print("input shape:", tuple(x.shape))
print("embedding output shape:", tuple(embeddings(x).shape))