# src/training/train_mistral.py
import os, yaml
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          Trainer, TrainingArguments, DataCollatorForLanguageModeling)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch
from src.utils.data_utils import build_samples

def load_config(cfg_path="configs/config_mistral.yaml"):
  with open(cfg_path, "r") as f:
    return yaml.safe_load(f)

def format_supervised(sample):
  # Format kiểu SFT: input + target nối lại để mô hình học sinh ra target
  # Ta sẽ dùng labels mask để chỉ tính loss trên phần summary
  return {"text": sample["prompt"] + "\n " + sample["summary"]}

def tokenize_func(tokenizer, max_len):
  def _tok(batch):
    enc = tokenizer(batch["text"], truncation=True, max_length=max_len)
    return enc
  return _tok

def make_labels_mask(tokenizer, batch_texts):
  # mask loss cho phần prompt: set label = -100
  inputs = tokenizer(batch_texts, truncation=True, max_length=tokenizer.model_max_length)
  labels = inputs["input_ids"][:]
  # heuristic: coi phần sau cùng (summary) là sau token "Summary:" hoặc newline cuối cùng
  # Đơn giản: không mask (cho dự án nhỏ), hoặc bạn có thể dùng special separator.
  return inputs, labels

def main():
  cfg = load_config()

  model_name = cfg["model"]["name"]
  max_len = cfg["model"]["max_length"]
  prompt_template = cfg["data"]["prompt_template"]
  input_field = cfg["data"]["input_field"]
  target_field = cfg["data"]["target_field"]
  max_input_sentences = cfg["data"]["max_input_sentences"]
  max_target_sentences = cfg["data"]["max_target_sentences"]

  train_samples = list(build_samples(
    cfg["data"]["train_path"], input_field, target_field,
    prompt_template, max_input_sentences, max_target_sentences
  ))
  val_samples = list(build_samples(
    cfg["data"]["val_path"], input_field, target_field,
    prompt_template, max_input_sentences, max_target_sentences
  ))

  train_ds = Dataset.from_list([format_supervised(s) for s in train_samples])
  val_ds   = Dataset.from_list([format_supervised(s) for s in val_samples])

  tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
  if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

  tokenized_train = train_ds.map(tokenize_func(tokenizer, max_len), batched=True, remove_columns=["text"])
  tokenized_val   = val_ds.map(tokenize_func(tokenizer, max_len), batched=True, remove_columns=["text"])

  model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
  )

  # LoRA
  lcfg = LoraConfig(
    r=cfg["lora"]["r"],
    lora_alpha=cfg["lora"]["alpha"],
    lora_dropout=cfg["lora"]["dropout"],
    target_modules=cfg["lora"]["target_modules"],
    bias="none",
    task_type="CAUSAL_LM",
  )
  model = get_peft_model(model, lcfg)

  args = TrainingArguments(
    output_dir=cfg["training"]["output_dir"],
    per_device_train_batch_size=cfg["training"]["per_device_train_batch_size"],
    per_device_eval_batch_size=cfg["training"]["per_device_eval_batch_size"],
    gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
    num_train_epochs=cfg["training"]["num_train_epochs"],
    learning_rate=cfg["training"]["learning_rate"],
    weight_decay=cfg["training"]["weight_decay"],
    warmup_ratio=cfg["training"]["warmup_ratio"],
    logging_steps=cfg["training"]["logging_steps"],
    save_steps=cfg["training"]["save_steps"],
    evaluation_strategy=cfg["training"]["evaluation_strategy"],
    eval_steps=cfg["training"]["eval_steps"],
    fp16=cfg["training"]["fp16"],
    lr_scheduler_type=cfg["training"]["lr_scheduler_type"],
    max_grad_norm=cfg["training"]["max_grad_norm"],
    report_to=[],
  )

  collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

  trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    data_collator=collator,
  )

  trainer.train()
  trainer.save_model(cfg["training"]["output_dir"])
  tokenizer.save_pretrained(cfg["training"]["output_dir"])

if __name__ == "__main__":
  main()
