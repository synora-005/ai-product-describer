"""Command-line entry point: `python -m app.cli --help`."""
import argparse
import os

DEFAULT_CORPUS = "data/products.txt"
DEFAULT_MODEL = "data/model.pt"
DEFAULT_TOKENIZER = "data/tokenizer.json"


def add_model_options(parser):
    parser.add_argument("--model", default=DEFAULT_MODEL, help="trained model file")
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER, help="tokenizer file")


def add_writing_options(parser):
    add_model_options(parser)
    parser.add_argument("--variants", type=int, default=1, help="how many different descriptions to write")
    parser.add_argument("--max-tokens", type=int, default=60, help="longest description, in tokens")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="0 = always the likeliest word; higher = more creative but messier")
    parser.add_argument("--top-k", type=int, default=40,
                        help="only pick from the k likeliest tokens (0 = no limit)")
    parser.add_argument("--seed", type=int, default=None, help="use the same number to repeat the same text")


def load_writer(args):
    for path in (args.model, args.tokenizer):
        if not os.path.exists(path):
            raise SystemExit(f"Not found: {path}. Train a model first with: python -m app.cli train")
    from app.core import BPETokenizer, load_model
    return load_model(args.model), BPETokenizer.load(args.tokenizer)


def write_descriptions(details, model, tokenizer, args):
    from app.core import describe_product
    if args.variants < 1:
        raise SystemExit("--variants must be at least 1.")
    try:
        texts = describe_product(
            model, tokenizer, details, variants=args.variants, max_new_tokens=args.max_tokens,
            temperature=args.temperature, top_k=args.top_k or None, seed=args.seed,
        )
    except ValueError as error:
        raise SystemExit(str(error))

    for number, text in enumerate(texts, start=1):
        print(text if len(texts) == 1 else f"{number}. {text}")

    save_path = getattr(args, "save", None)
    if save_path:
        with open(save_path, "a", encoding="utf-8") as f:
            for text in texts:
                f.write(" ".join(text.split()) + "\n")
        print(f"(saved to {save_path})")


def run_train(args):
    from app.core import train_from_corpus
    if not os.path.exists(args.corpus):
        raise SystemExit(f"Corpus not found: {args.corpus}. See the README for how to build it.")

    def show(record):
        line = f"step {record['step']:>5}   train loss {record['train_loss']:.3f}"
        if "val_loss" in record:
            line += f"   val loss {record['val_loss']:.3f}"
        print(line)

    try:
        train_from_corpus(args.corpus, args.model, args.tokenizer, vocab_size=args.vocab_size,
                          steps=args.steps, eval_every=args.eval_every, log=show)
    except ValueError as error:
        raise SystemExit(str(error))
    print(f"Saved {args.model} and {args.tokenizer}")


def run_interactive(args):
    model, tokenizer = load_writer(args)
    print("Type product details, like: oversized cotton t-shirt. Press Enter on an empty line to quit.")
    while True:
        try:
            details = input("Product details> ").strip()
        except EOFError:
            break
        if not details:
            break
        write_descriptions(details, model, tokenizer, args)
        print()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="ai-product-describer", description="Small business owners use this CLI tool to generate creative product descriptions for their inventory by training a custom GPT on examples of existing product texts.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("hello", help="check that everything is wired up")

    train = sub.add_parser("train", help="learn the tokenizer and train the model from a corpus")
    train.add_argument("--corpus", default=DEFAULT_CORPUS, help="text file, one description per line")
    train.add_argument("--steps", type=int, default=1000, help="number of training steps")
    train.add_argument("--eval-every", type=int, default=100, help="print the losses this often")
    train.add_argument("--vocab-size", type=int, default=512, help="tokenizer vocabulary size")
    add_model_options(train)

    describe = sub.add_parser("describe", help="write a product description from product details")
    describe.add_argument("details", nargs="*", help="what you know about the product, e.g. oversized cotton t-shirt")
    describe.add_argument("--save", default=None, help="also append the descriptions to this text file")
    add_writing_options(describe)

    interactive = sub.add_parser("interactive", help="keep typing product details and get descriptions")
    add_writing_options(interactive)

    args = parser.parse_args(argv)
    if args.command == "hello":
        print("AI Product Describer is ready. Open README.md for your milestones.")
    elif args.command == "train":
        run_train(args)
    elif args.command == "describe":
        model, tokenizer = load_writer(args)
        write_descriptions(args.details, model, tokenizer, args)
    elif args.command == "interactive":
        run_interactive(args)


if __name__ == "__main__":
    main()