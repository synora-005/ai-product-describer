import torch

from app.core import BPETokenizer, Embeddings, TransformerBlock, encode_corpus, load_corpus

descriptions = load_corpus("data/products.txt")
tokenizer = BPETokenizer.load("data/tokenizer.json")   # saved in milestone 2
all_ids = encode_corpus(descriptions, tokenizer)

torch.manual_seed(0)
embeddings = Embeddings(tokenizer.vocab_size, block_size=64, n_embd=32)
block = TransformerBlock(n_embd=32, n_head=4, block_size=64)

x = torch.tensor(all_ids[:64]).unsqueeze(0)
h = embeddings(x)
out = block(h)
print("in:", tuple(h.shape), "out:", tuple(out.shape))
print("block parameters:", sum(p.numel() for p in block.parameters()))

# causal check: the first 10 outputs must not depend on the tokens after them
out_short = block(h[:, :10])
print("causal check:", torch.allclose(out[:, :10], out_short, atol=1e-5))