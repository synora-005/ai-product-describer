from app.core import load_corpus, corpus_stats

descriptions = load_corpus("data/products.txt")
print(corpus_stats(descriptions))
for d in descriptions[:3]:
    print("-", d)