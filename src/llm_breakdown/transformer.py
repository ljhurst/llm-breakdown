import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, GPT2Tokenizer


def demo() -> None:
    model_name = "gpt2"

    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    _model_config(model)

    text = "A wise man once said: Penguins are cute."

    tokens = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**tokens, output_hidden_states=True)

    _hidden_states(tokens, outputs)
    _cosine_similarity_for_token_between_layers(tokens, outputs, model)
    _manipulate_hidden_states(tokenizer, model)


def _model_config(model: AutoModelForCausalLM) -> None:
    print("Model configuration:")
    print(model.config)
    print()


def _hidden_states(tokens: dict[str, torch.Tensor], outputs) -> None:
    print("Hidden states:")
    for key, value in tokens.items():
        print(f"'{key}' contains:")
        print(" ", value)
        print()
    print()

    print("Keys in 'outputs':", outputs.keys())
    print("Size of outputs.logits:", outputs.logits.shape)
    print("Number of hidden states:", len(outputs.hidden_states))
    print("Size of each hidden state:", outputs.hidden_states[0].shape)
    print()


def _cosine_similarity_for_token_between_layers(
    tokens: dict[str, torch.Tensor],
    outputs,
    model: AutoModelForCausalLM,
) -> None:
    print("Cosine similarity of hidden states between layers:")
    hs = outputs.hidden_states
    num_hidden = len(hs)
    hidden_dim = model.config.n_embd

    tokens_to_analyze = np.linspace(
        0, len(tokens["input_ids"][0]) - 1, num=4, dtype=int
    )

    for token in tokens_to_analyze:
        all_hiddens = torch.zeros((num_hidden, hidden_dim))

        for layer in range(num_hidden):
            all_hiddens[layer, :] = hs[layer][0, token, :]

        cos_sim = F.cosine_similarity(
            all_hiddens.unsqueeze(0), all_hiddens.unsqueeze(1), dim=-1
        )

        print(f"Cosine similarity between hidden states for token {token}:")
        print(cos_sim)
    print()


def _manipulate_hidden_states(
    tokenizer: GPT2Tokenizer, model: AutoModelForCausalLM
) -> None:
    print("Manipulating hidden states:")
    txt = "As Gregor Samsa awoke one morning from uneasy dreams, he found himself transformed in his bed into a gigantic"

    tokens = tokenizer(txt, return_tensors="pt")

    print("The text contains:")
    print(f"  {len(txt)} characters ({len(set(txt))} unique)")
    print(
        f"  {len(tokens['input_ids'][0])} tokens ({len(set(tokens['input_ids'][0]))} unique)"
    )

    with torch.no_grad():
        outputs = model(**tokens, output_hidden_states=True)

    _, indices = torch.topk(outputs.logits[0, -1, :], k=21)

    print("Top 21 possible next words:")
    for idx in indices:
        print(f"  '{tokenizer.decode(idx)}'")
    print()

    target_token_idx = tokenizer.encode(" insect")[0]

    log_sm_logits = F.log_softmax(outputs.logits[0, -1, :], dim=-1)
    target_logsm_clean = log_sm_logits[target_token_idx]

    print(f"Original log-softmax for target token is {target_logsm_clean:.6f}")

    num_hidden = len(outputs.hidden_states)

    target_logsm = np.zeros(num_hidden - 1)

    for layer in range(num_hidden - 1):

        def hook_fn(module, input, output):
            hidden, *rest = output
            hidden.mul_(0.8)

            return tuple([hidden] + rest)

        hook_handle = model.transformer.h[layer].register_forward_hook(hook_fn)

        with torch.no_grad():
            outputs = model(**tokens, output_hidden_states=True)

        hook_handle.remove()

        log_sm_logits = F.log_softmax(outputs.logits[0, -1, :], dim=-1)
        target_logsm[layer] = log_sm_logits[target_token_idx]

        print(
            f"After manipulating layer {layer}, log-softmax for target token is {target_logsm[layer]:.6f}"
        )
