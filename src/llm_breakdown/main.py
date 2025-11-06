from llm_breakdown import attention, embeddings, logits, mlp, tokenization, transformer


def main() -> None:
    print("Starting Tokenization Demo...")
    print("----------------------------------")
    tokenization.demo()

    print("Starting Logits Demo...")
    print("----------------------------------")
    logits.demo()

    print("Starting Embeddings Demo...")
    print("----------------------------------")
    embeddings.demo()

    print("Starting Transformers Demo...")
    print("----------------------------------")
    transformer.demo()

    print("Starting Attention Demo...")
    print("----------------------------------")
    attention.demo()

    print("Starting MLP Demo...")
    print("----------------------------------")
    mlp.demo()


if __name__ == "__main__":
    main()
