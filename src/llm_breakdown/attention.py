import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, GPT2Tokenizer


def demo() -> None:
    model_name = "gpt2-large"

    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    activations = {}

    def implant_hook(layer_number: int):
        def hook(module, input, output):
            activations[f"L{layer_number}_qvk"] = output.detach()

        return hook

    hook_handles = []

    for layer_idx in range(model.config.n_layer):
        layer_name = model.transformer.h[layer_idx].attn.c_attn
        hook_handles.append(layer_name.register_forward_hook(implant_hook(layer_idx)))

    text = "Be who you are and say what you feel, because those who mind don't matter, and those who matter don't mind"
    tokens = tokenizer(text, return_tensors="pt")

    n_tokens = len(tokens["input_ids"][0])

    with torch.no_grad():
        _ = model(**tokens)

    for handle in hook_handles:
        handle.remove()

    print("Activation keys and shapes:")
    print(activations.keys())
    print(activations["L5_qvk"].shape)
    print()

    q, k, v = torch.split(activations["L5_qvk"], model.config.n_embd, dim=-1)
    print("Q, K, V shapes:")
    print("Q shape:", q.shape)
    print("K shape:", k.shape)
    print("V shape:", v.shape)
    print()

    print("Computing Q @ K^T for the second token in the sequence:")
    qkt = q[0, 1:, :] @ k[0, 1:, :].transpose(-2, -1)
    print("Q @ K^T shape:", qkt.shape)
    print()

    n_layers = model.config.n_layer
    n_embd = model.config.n_embd
    n_heads = model.config.n_head

    head_dim = n_embd // n_heads

    q, k, v = torch.split(activations["L9_qvk"][0, :, :], n_embd, dim=1)

    q_h = torch.split(q, head_dim, dim=1)

    print(f"There are {len(q_h)} heads")
    print(f"Each head has size {q_h[2].shape}")
    print()

    ablation_logits = np.zeros(n_layers)

    print("Ablating head 4 in each layer and measuring logit for the next token:")
    for layer in range(n_layers):

        def hook_to_ablate(module, input):
            head_tensor = input[0].view(1, n_tokens, n_heads, head_dim)

            head_tensor[:, -2, 4, :] = 0

            head_tensor = head_tensor.view(1, n_tokens, n_embd)

            return tuple(head_tensor, *input[1:])

        layer_to_implant = model.transformer.h[layer].attn.c_proj
        handle = layer_to_implant.register_forward_pre_hook(hook_to_ablate)

        with torch.no_grad():
            outputs_ablated = model(**tokens)

        handle.remove()

        log_smax = F.log_softmax(outputs_ablated.logits[0, -2, :], dim=-1)
        ablation_logits[layer] = log_smax[tokens["input_ids"][0, -1]]

        print(
            f"After ablating head 4 in layer {layer}, log-softmax for next token is {ablation_logits[layer]:.6f}"
        )
    print()
