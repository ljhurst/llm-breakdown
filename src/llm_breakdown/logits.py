import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, GPT2Tokenizer


def demo() -> None:
    model_name = "gpt2"

    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    _logits_and_softmax(tokenizer, model)

    tokens = tokenizer.encode("I like oat milk in my", return_tensors="pt")
    final_logits = model(tokens).logits[0, -1, :].detach()

    _most_likely_tokens(tokenizer, final_logits)
    _coffee_rank(tokenizer, final_logits)
    _multinomial_selection(tokenizer, final_logits)

    _generate_tokens_loop(tokenizer, model)
    _generate_tokens(tokenizer, model)

    _manipulate_coffee(tokenizer, model)


def _logits_and_softmax(tokenizer: GPT2Tokenizer, model: AutoModelForCausalLM) -> None:
    txt = "I think more people would eat tumeric if it were purple."

    tokens = tokenizer(txt, return_tensors="pt")
    print(f"There are {len(txt)} characters and {len(tokens['input_ids'][0])} tokens.")

    model.eval()
    with torch.no_grad():
        output = model(tokens["input_ids"])

    print("Logits shape:")
    print(output.logits.shape)
    print()

    logits = output.logits[0, 4, :].detach()

    softmax_direct = torch.exp(logits) / torch.exp(logits).sum()
    softmax_logits = F.softmax(logits, dim=-1)

    print("Are the two softmax implementations equal?")
    print("Softmax direct:", softmax_direct)
    print("Softmax logits:", softmax_logits)
    print()


def _most_likely_tokens(tokenizer: GPT2Tokenizer, logits: torch.Tensor) -> None:
    max_logit = torch.argmax(logits)
    print(f"The most likely next token is: '{tokenizer.decode(max_logit)}'")
    print()

    print("   Logit   |     Token")
    print("-----------+-----------")
    for token in torch.topk(logits, k=10)[1]:
        print(f"{logits[token]:.3f} | '{tokenizer.decode(token)}'")
    print()


def _coffee_rank(tokenizer: GPT2Tokenizer, logits: torch.Tensor) -> None:
    coffee_idx = tokenizer.encode(" coffee")[0]
    print(f"' coffee' has index {coffee_idx}")

    sidx = torch.argsort(logits, descending=True)
    rank = torch.where(sidx == coffee_idx)[0]
    print(f"' coffee' is ranked {rank.item() + 1} in likelihood among all tokens.")
    print()


def _multinomial_selection(tokenizer: GPT2Tokenizer, logits: torch.Tensor) -> None:
    print("Using multinomial sampling:")
    softmax_logits = F.softmax(logits, dim=-1)
    multinomial_tokens = torch.multinomial(softmax_logits, num_samples=5)

    for token in multinomial_tokens:
        print(f"'{tokenizer.decode(token)}'")
    print()


def _generate_tokens_loop(
    tokenizer: GPT2Tokenizer, model: AutoModelForCausalLM
) -> None:
    print("Generating tokens using manual loop:")
    sentence = "I like oat milk in my cereal"
    tokens = tokenizer.encode(sentence, return_tensors="pt")

    for i in range(10):
        with torch.no_grad():
            logits = model(tokens).logits[0, -1, :]

        softmax_logits = F.softmax(logits, dim=-1)

        next_token = torch.multinomial(softmax_logits, num_samples=1)

        tokens = torch.cat([tokens, torch.tensor([[next_token]])], dim=-1)

        print(f"Iteration {i + 1}: {tokenizer.decode(tokens[0])}")
    print()


def _generate_tokens(tokenizer: GPT2Tokenizer, model: AutoModelForCausalLM) -> str:
    print("Generating tokens using model.generate():")
    tokens = tokenizer.encode("I like oat milk in my", return_tensors="pt")

    token_seq = model.generate(tokens, max_new_tokens=10, do_sample=True)
    print(tokenizer.decode(token_seq[0]))
    print()


def _manipulate_coffee(tokenizer: GPT2Tokenizer, model: AutoModelForCausalLM) -> None:
    print("Manipulating logits to favor ' coffee':")

    tokens = tokenizer.encode("I like oat milk in my", return_tensors="pt")
    coffee_idx = tokenizer.encode(" coffee")[0]

    def hook(module, input, output):
        actual_max = torch.argmax(output[0, -1, :])
        output[0, -1, coffee_idx] = output[0, -1, actual_max] + 10

        return output

    hook_handle = model.lm_head.register_forward_hook(hook)

    final_logits = model(tokens).logits[0, -1, :].detach()
    max_logit = torch.argmax(final_logits)
    print(f"Most likely token next token is '{tokenizer.decode(max_logit)}'")
    print()

    hook_handle.remove()
