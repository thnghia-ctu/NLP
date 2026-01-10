import os
import yaml
import torch
from pathlib import Path
from typing import Dict, Any

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model

from src.preprocessing.build_train_data import build_pts_samples


# ----------------- CONFIG -----------------
def load_config(config_path: str = "configs/config_mistral.yaml") -> Dict[str, Any]:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_text_field(example: Dict, field_name: str, fallback_keys: list = None) -> str:
    value = example.get(field_name, "")

    if isinstance(value, str):
        return value

    if isinstance(value, dict):
        for k in ["text", "content", "section_text", "summary_text", "value"]:
            if k in value:
                return value[k]

    if fallback_keys:
        for fk in fallback_keys:
            if fk in example:
                v = example[fk]
                return v if isinstance(v, str) else ""

    return ""


def format_example(example: Dict, tokenizer) -> Dict:
    prompt = get_text_field(example, "prompt", ["instruction", "question", "input"])
    if not prompt:
        raise ValueError(f"Không tìm thấy prompt trong sample: {example.keys()}")

    summary = ""
    for key in ["summary", "summary_text", "completion", "response", "output", "target", "answer"]:
        summary = get_text_field(example, key)
        if summary:
            break

    if not summary:
        raise ValueError(f"Không tìm thấy response trong sample: {example.keys()}")

    text = f"{prompt.strip()}\n\n### Tóm tắt:\n{summary.strip()}{tokenizer.eos_token}"
    return {"text": text}


# ----------------- MAIN -----------------
def main():
    cfg = load_config()
    model_name = cfg["model"]["name"]
    output_dir = Path(cfg["training"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Training model: {model_name}")
    print(f"Output dir: {output_dir}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load dataset
    train_samples = list(build_pts_samples(cfg["data"]["train_path"]))
    val_samples = list(build_pts_samples(cfg["data"]["val_path"]))

    train_dataset = Dataset.from_list([format_example(s, tokenizer) for s in train_samples])
    val_dataset = Dataset.from_list([format_example(s, tokenizer) for s in val_samples])

    # Tokenize
    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, max_length=cfg["model"]["max_length"])

    tokenized_train = train_dataset.map(tokenize_fn, batched=True, num_proc=4, remove_columns=train_dataset.column_names)
    tokenized_val = val_dataset.map(tokenize_fn, batched=True, num_proc=4, remove_columns=val_dataset.column_names)

    # Load FP16 model (no bitsandbytes, no Triton)
    print("Loading FP16 model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    # LoRA
    peft_config = LoraConfig(
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        target_modules=cfg["lora"]["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Training args
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        per_device_train_batch_size=cfg["training"]["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["training"]["per_device_eval_batch_size"],
        gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
        learning_rate=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
        warmup_ratio=cfg["training"]["warmup_ratio"],
        lr_scheduler_type=cfg["training"]["lr_scheduler_type"],
        logging_steps=cfg["training"]["logging_steps"],
        save_strategy="steps",
        save_steps=cfg["training"]["save_steps"],
        eval_strategy="steps",
        eval_steps=cfg["training"]["eval_steps"],
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=True,
        optim="adamw_torch",
        report_to="none",
        seed=42,
    )

    # Trainer
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    print("Start training...")
    trainer.train()

    final_path = output_dir / "final_adapter"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    print(f"Done! Adapter saved at {final_path}")


if __name__ == "__main__":
    main()
