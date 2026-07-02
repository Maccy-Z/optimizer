import torch
from llm import NemotronHForCausalLM
from transformers import AutoTokenizer

MODEL_NAME = "nvidia/NVIDIA-Nemotron-Nano-9B-v2"


def hook(w):
    print(f'{w.grad = }')
    w.grad = None
    return


def setup_hooks(model):
    for n, p in model.named_parameters():
        p.register_post_accumulate_grad_hook(hook)


def calculate_loss(model: NemotronHForCausalLM, text, tokenizer, device):
    inputs = tokenizer(text, return_tensors="pt").to(device)
    print(f'{inputs["input_ids"].shape = }')
    # For causal LM training, labels are usually the same token IDs as inputs.
    # The model shifts them internally when computing next-token loss.
    labels = inputs["input_ids"].clone()

    outputs = model(**inputs, labels=labels)
    return outputs.loss


def main():
    prompt = """Autonomous artificial intelligence agents risk causing “market meltdown” and may need tighter regulation, Bank of England Deputy Governor Sarah Breeden warned.

Speaking in Sintra, Portugal, Breeden said that agents “could amplify volatility in stress” in financial markets if they all respond in the same way to similar prompts.

While she said investors largely use AI for lower-risk tasks such as research currently, the use of agents that perform tasks autonomously could rise rapidly. Regulators also need to closely watch what these agents mean for consumers and payments.

“What these two examples – agentic commerce and agentic trading – both highlight is that, as AI capabilities increase, we must keep asking whether existing, technology-agnostic regulatory frameworks remain sufficient,” she said at a European Central Bank conference.

As companies increasingly adopt AI models, the use of agents is seen as one of the key ways the technology can help boost productivity by doing tasks on their own but guided by humans.

For consumers, this could mean AI booking holidays or refilling their fridges using the agents, according to Breeden. The finance sector could use agents to execute trading strategies.

“If AI agents respond similarly to the same prompts or triggers, they could amplify volatility in stress – especially if their objectives drift from original goals or public policy objectives, in a manifestation of the misalignment problem that can arise with some AI models,” Breeden said.
"""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = NemotronHForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    # setup_hooks(model)
    model.eval()

    loss = calculate_loss(model, prompt, tokenizer, device)
    print(f"Dummy training loss: {loss.item():.4f}")

    # loss.backward()

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.inference_mode():
        print("Starting inference: " )
        output_ids = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(response)


if __name__ == "__main__":
    main()
