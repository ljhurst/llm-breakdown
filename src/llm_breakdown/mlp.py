import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, GPT2Tokenizer


def demo() -> None:
    model_name = "gpt2-medium"

    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    n_layers = model.config.n_layer
    n_embd = model.config.n_embd

    activations = {}

    def make_mlp_hook(layer_index: int):
        def hook(module, inputs, output):
            activations[f"1_L{layer_index}"] = inputs[0].detach()
            activations[f"2_L{layer_index}"] = module.c_fc(inputs[0]).detach()
            activations[f"3_L{layer_index}"] = F.gelu(activations[f"2_L{layer_index}"])
            activations[f"4_L{layer_index}"] = output.detach()

        return hook

    hook_handles = []

    for layer in range(n_layers):
        module_name = model.transformer.h[layer].mlp
        handle = module_name.register_forward_hook(make_mlp_hook(layer))
        hook_handles.append(handle)

    sentences = [
        "If the sun is round, then the moon is round.",
        "If a square is a square, then why isn't a triangle square?",
    ]

    tokenizer.pad_token = tokenizer.eos_token

    tokens = tokenizer(sentences, return_tensors="pt", padding=True)

    for index, sentence in enumerate(sentences):
        print("Sentence:", index)
        print(" ", tokens)
    print()

    tokens1 = torch.where(tokens["attention_mask"][0])[0].tolist()[1:]
    tokens2 = torch.where(tokens["attention_mask"][1])[0].tolist()[1:]

    tokens_to_use = [tokens1, tokens2]
    print("Tokens to use:", tokens_to_use)
    print()

    with torch.no_grad():
        outputs = model(**tokens)

    for handle in hook_handles:
        handle.remove()

    print("MLP Activations:")
    print(activations.keys())
    print()

    print("1_L0 shape:", activations["1_L10"].shape)
    print("2_L0 shape:", activations["2_L10"].shape)
    print("3_L0 shape:", activations["3_L10"].shape)
    print("4_L0 shape:", activations["4_L10"].shape)
    print()

    binedges = np.linspace(-3, 2, 201)

    for i in range(1, 5):
        all_activations = np.concatenate(
            (
                activations[f"{i}_L10"][0, tokens_to_use[0], :].flatten(),
                activations[f"{i}_L10"][0, tokens_to_use[0], :].flatten(),
            ),
            axis=0,
        )

        y, x = np.histogram(all_activations, bins=binedges)

        print(f"Activation {i} histogram:")
        for bin_left, count in zip(x[:-1], y):
            print(f"  Bin starting at {bin_left:.2f}: {count}")
        print()

    text = "It was dark and stormy"
    target_index = tokenizer.encode(" night")[0]

    tokens = tokenizer(text, return_tensors="pt")

    target_logprobs = np.zeros(n_layers + 1)

    pct_ablation = 0.08

    print(
        "Ablating top "
        f"{int(pct_ablation * 100)}% of MLP neurons in each layer and measuring logprob of ' night':"
    )
    for layer in range(-1, n_layers):

        def replace_hook(module, input, output):
            idx = torch.topk(output[0, -1, :], int(pct_ablation * n_embd)).indices

            output[0, -1, idx] = torch.mean(output[0, -1, :])

            return output

        if layer > -1:
            handle = model.transformer.h[layer].mlp.c_proj.register_forward_hook(
                replace_hook
            )

        with torch.no_grad():
            outputs = model(**tokens)

        if layer > -1:
            handle.remove()

        target_logprobs[layer + 1] = F.log_softmax(
            outputs.logits[0, -1, :].detach(), dim=-1
        )[target_index]
        print(
            f"After ablating layer {layer}, log-probability for target token is {target_logprobs[layer + 1]:.6f}"
        )
