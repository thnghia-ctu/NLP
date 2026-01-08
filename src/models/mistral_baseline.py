import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float16
)

def summarize_article(article_text, max_input_tokens=4096, max_new_tokens=256):
    prompt = f"""
Summarize the following scientific article into a concise abstract:

{article_text}
"""

    inputs = tokenizer(
        prompt,
        truncation=True,
        max_length=max_input_tokens,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False
        )

    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary
