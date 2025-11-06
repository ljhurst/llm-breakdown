import numpy as np
from transformers import BertTokenizer, GPT2Tokenizer, PreTrainedTokenizer


def demo() -> None:
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    _vocab_size(tokenizer)
    _example_tokens(tokenizer)
    _example_encoding(tokenizer)

    print("GPT-2 Tokenization:")
    _encoding_comparision(tokenizer)

    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    print("BERT Tokenization:")
    _encoding_comparision(bert_tokenizer)


def _vocab_size(tokenizer: GPT2Tokenizer) -> None:
    print("Vocab Size:", tokenizer.vocab_size)
    print()


def _example_tokens(tokenizer: GPT2Tokenizer) -> None:
    print("Example Tokens:")
    for token in np.random.randint(tokenizer.vocab_size, size=10):
        print(f"Token ID: {token:t>5} -> Token: '{tokenizer.decode([token])}'")
    print()


def _example_encoding(tokenizer: PreTrainedTokenizer) -> None:
    txt = "I like the longer-form posts on Substack."

    tokens = tokenizer.encode(txt)
    print(f"The sentence contains {len(txt)} characters and {len(tokens)} tokens.")
    print()

    print("Token ID | Token")
    print("---------+---------")

    for token in tokens:
        print(f" {token:7d} | '{tokenizer.decode([token])}'")
    print()


def _encoding_comparision(tokenizer: PreTrainedTokenizer) -> None:
    words = [
        "banana",
        " banana",
        " Banana",
        "substack",
        " substack",
        "like",
        " like",
        ".",
        ",",
        " ",
    ]

    for word in words:
        token_ids = tokenizer.encode(word)
        print(f"{len(token_ids)} tokens form '{word}' ({token_ids})")
    print()
