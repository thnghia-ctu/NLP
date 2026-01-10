import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
# Đổi sang Ministral 3B
BASE_MODEL = "mistralai/Ministral-3B-Instruct-v0.1"
LORA_PATH = r"experiments/checkpoints/mistral_lora_arxiv/final_adapter"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
base_model  = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="auto",
    torch_dtype=torch.float16
)

model = PeftModel.from_pretrained(
    base_model,
    LORA_PATH,
    torch_dtype=torch.float16
)

model.eval()

model = model.merge_and_unload()

def summarize_section(
    section_text,
    max_input_tokens=1024,
    max_new_tokens=80
):
    prompt = (
        "Summarize the following SECTION of a scientific paper "
        "in 2–3 concise sentences. "
        "Focus only on key technical points.\n\n"
        f"{section_text}"
    )

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
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    # ✨ CẮT PROMPT
    gen_ids = outputs[0][inputs["input_ids"].shape[-1]:]
    summary = tokenizer.decode(gen_ids, skip_special_tokens=True)

    return summary.strip()

def merge_section_summaries(section_summaries):
    text = "\n".join(
        f"- {s}" for s in section_summaries
    )

    prompt = (
        "You are given summaries of sections of a scientific paper.\n"
        "Write a coherent abstract combining them.\n\n"
        f"{text}"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False
        )

    gen_ids = outputs[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True)
