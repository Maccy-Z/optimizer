import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch
from transformers import AutoTokenizer

from utils import print_max_memory
from llm import NemotronHForCausalLM

MODEL_NAME = "nvidia/NVIDIA-Nemotron-Nano-9B-v2"
MAX_TRAIN_TOKENS = 300


def hook(w):
    w.grad = None
    return


def setup_hooks(model):
    for n, p in model.named_parameters():
        p.register_post_accumulate_grad_hook(hook)


def calculate_loss(model: NemotronHForCausalLM, text, tokenizer, device, max_tokens=MAX_TRAIN_TOKENS):
    tokenizer_kwargs = {"return_tensors": "pt"}
    if max_tokens is not None:
        tokenizer_kwargs.update(
            {
                "max_length": max_tokens,
                "truncation": True,
            }
        )

    inputs = tokenizer(text, **tokenizer_kwargs).to(device)
    print(f"{max_tokens = }")
    print(f'{inputs["input_ids"].shape = }')
    # For causal LM training, labels are usually the same token IDs as inputs.
    # The model shifts them internally when computing next-token loss.
    labels = inputs["input_ids"].clone()

    outputs = model(**inputs, labels=labels, use_cache=False)
    return outputs.loss


def main():
    prompt = """Autonomous artificial intelligence agents risk causing “market meltdown” and may need tighter regulation, Bank of England Deputy Governor Sarah Breeden warned.

Speaking in Sintra, Portugal, Breeden said that agents “could amplify volatility in stress” in financial markets if they all respond in the same way to similar prompts.

While she said investors largely use AI for lower-risk tasks such as research currently, the use of agents that perform tasks autonomously could rise rapidly. Regulators also need to closely watch what these agents mean for consumers and payments.

“What these two examples – agentic commerce and agentic trading – both highlight is that, as AI capabilities increase, we must keep asking whether existing, technology-agnostic regulatory frameworks remain sufficient,” she said at a European Central Bank conference.

As companies increasingly adopt AI models, the use of agents is seen as one of the key ways the technology can help boost productivity by doing tasks on their own but guided by humans.

For consumers, this could mean AI booking holidays or refilling their fridges using the agents, according to Breeden. The finance sector could use agents to execute trading strategies.

“If AI agents respond similarly to the same prompts or triggers, they could amplify volatility in stress – especially if their objectives drift from original goals or public policy objectives, in a manifestation of the misalignment problem that can arise with some AI models,” Breeden said.

The UK central bank is working with the Bank for International Settlements and Bundesbank on understanding whether agents can drive “herding behavior” and how officials can tackle the problem. This includes whether to implement guardrails such as “circuit breakers or kill switches that would limit or stop trading market-wide if faulty AI models cause market meltdown.”

In a question-and-answer session, Breeden likened AI models to teenagers in that “they lie, they tell you they’ve not done things when they have and they behave differently when you’re watching them.” She added that it will be key for regulators to have a human who is “accountable for that model.”

BOE Governor Andrew Bailey has said the technology could help revive Britain’s tepid economic growth rates. However, UK officials are concerned about its effect on the banking sector, particularly if new models expose cyber vulnerabilities across the economy.

Breeden said international cooperation will be crucial, warning that new AI capabilities can spread across borders through common technology dependencies, globally systemic financial institutions and market infrastructure.

“What if the next surprise puts the latest capabilities in bad actors’ hands, or models develop in ways that are harder to evaluate and control, given evidence that some behave differently in testing compared to real-life scenarios?” she said. “We shouldn’t wait for a crisis to build the cooperation we need.”
"""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = NemotronHForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    setup_hooks(model)

    model.train()

    loss = calculate_loss(model, prompt, tokenizer, device, max_tokens=MAX_TRAIN_TOKENS)
    print_max_memory("After forward pass")

    print(f"Dummy training loss: {loss.item():.4f}")
    loss.backward()
    print_max_memory("After backward pass")

    # inputs = tokenizer(prompt, return_tensors="pt").to(device)
    #
    # with torch.inference_mode():
    #     print("Starting inference: " )
    #     output_ids = model.generate(
    #         **inputs,
    #         max_new_tokens=80,
    #         do_sample=False,
    #         pad_token_id=tokenizer.eos_token_id,
    #     )
    #
    # response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    # print(response)


if __name__ == "__main__":
    main()
