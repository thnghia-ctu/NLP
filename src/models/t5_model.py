import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

# Ví dụ: t5-small hoặc t5-base
MODEL_NAME = "t5-base"
ADAPTER_PATH = "experiments/checkpoints/t5_lora_arxiv/final_adapter"  # nếu có
USE_LORA = False  # đổi True nếu bạn có LoRA cho T5

_t5_model = None
_t5_tokenizer = None


def load_t5_model():
    global _t5_model, _t5_tokenizer

    if _t5_model is None:
        _t5_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        base_model = AutoModelForSeq2SeqLM.from_pretrained(
            MODEL_NAME,
            device_map="auto",
            torch_dtype=torch.float16
        )

        if USE_LORA:
            peft_model = PeftModel.from_pretrained(
                base_model,
                ADAPTER_PATH
            )
            _t5_model = peft_model.merge_and_unload()
        else:
            _t5_model = base_model

        _t5_model.eval()

    return _t5_model, _t5_tokenizer

def summarize_section_t5(
    section_text,
    max_input_tokens=1024,
    max_new_tokens=80
):
    """
    T5-based section summarization
    """

    #T5 BẮT BUỘC có task prefix
    input_text = (
        "summarize: "
        "Summarize the following SECTION of a scientific paper "
        "in 2–3 concise sentences, focusing on key technical points.\n\n"
        f"{section_text}"
    )

    model, tokenizer = load_t5_model()

    inputs = tokenizer(
        input_text,
        truncation=True,
        max_length=max_input_tokens,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=4,              # T5 hợp beam search
            early_stopping=True
        )

    summary = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return summary.strip()
