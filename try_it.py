from app.core import BPETokenizer, generate_description, load_model

tokenizer = BPETokenizer.load("data/tokenizer.json")
model = load_model("data/model.pt")

for prompt in ["", "Oversized cotton t-shirt", "Fitted top in", "V-neck"]:
    print(f"\nPrompt: {prompt!r}")
    print("  greedy :", generate_description(model, tokenizer, prompt, temperature=0))
    for temperature in (0.5, 0.8, 1.2):
        text = generate_description(model, tokenizer, prompt, temperature=temperature, seed=1)
        print(f"  T={temperature}:", text)