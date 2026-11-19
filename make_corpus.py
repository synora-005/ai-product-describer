import csv
import re

seen = set()
lines = []
with open("data/articles.csv", encoding="utf-8", newline="") as f:
    for row in csv.DictReader(f):
        if row["product_type_name"] != "T-shirt":
            continue
        text = re.sub(r"\s+", " ", row["detail_desc"]).strip()
        if len(text) < 30 or text in seen:
            continue
        seen.add(text)
        lines.append(text)

with open("data/products.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print(len(lines), "descriptions written")