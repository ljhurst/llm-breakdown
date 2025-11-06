import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, GPT2Tokenizer


def demo() -> None:
    model_name = "gpt2"

    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    _embeddings_shape(model)

    _manipulate_embeddings(tokenizer, model)


def _embeddings_shape(model: AutoModelForCausalLM) -> None:
    embeddings = model.transformer.wte.weight.detach()

    print("Embeddings shape:", embeddings.shape)
    print()


def _manipulate_embeddings(
    tokenizer: GPT2Tokenizer, model: AutoModelForCausalLM
) -> None:
    print("Manipulating embeddings...")
    text = "The capital of Germany is"

    source = " Germany"
    target = " Berlin"

    distractor_source = " France"
    distractor_target = " Paris"

    tokens = tokenizer(text, return_tensors="pt")

    source_idx = tokenizer.encode(source)
    target_idx = tokenizer.encode(target)

    distractor_source_idx = tokenizer.encode(distractor_source)
    distractor_target_idx = tokenizer.encode(distractor_target)

    country_loc = torch.where(tokens["input_ids"][0] == tokenizer.encode(source)[0])[
        0
    ].item()

    with torch.no_grad():
        sm_logits = F.softmax(model(tokens["input_ids"]).logits.detach(), dim=-1)

    target_logit = sm_logits[0, -1, target_idx]
    distractor_target_logit = sm_logits[0, -1, distractor_target_idx]

    print(f"Original logit for '{target}': {target_logit.item():.6f}")
    print(
        f"Original logit for '{distractor_target}': {distractor_target_logit.item():.6f}"
    )
    print()

    embeddings = model.transformer.wte.weight.detach()

    for p_germany in [0.0, 0.25, 0.5, 0.75, 1.0]:
        print(f"Mixing embeddings with p_germany = {p_germany:.2f}")

        def hook(module, input, output):
            print(f"Variable 'output' has shape: {output.shape}")

            mixed_vector = (
                p_germany * embeddings[source_idx, :]
                + (1 - p_germany) * embeddings[distractor_source_idx, :]
            )

            print("Variable 'mixed_vector' has shape:", mixed_vector.shape)
            output[0, country_loc, :] = mixed_vector

            return output

        handle = model.transformer.wte.register_forward_hook(hook)

        with torch.no_grad():
            sm_logits_modified = F.softmax(
                model(tokens["input_ids"]).logits.detach(), dim=-1
            )

        target_logit_modified = sm_logits_modified[0, -1, target_idx]
        distractor_target_logit_modified = sm_logits_modified[
            0, -1, distractor_target_idx
        ]

        print(f"Modified logit for '{target}': {target_logit_modified.item():.6f}")
        print(
            f"Modified logit for '{distractor_target}': {distractor_target_logit_modified.item():.6f}"
        )
        print()

        handle.remove()
