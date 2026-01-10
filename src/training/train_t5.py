import yaml
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)

from src.preprocessing.build_train_data import build_pts_samples


# --------------------------------------------------
# Load config
# --------------------------------------------------
def load_config():
    with open("configs/config_t5.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


# --------------------------------------------------
# Format data for T5 (instruction-style)
# --------------------------------------------------
def format_t5(sample):
    return {
        "input_text": (
            "summarize scientific section:\n"
            + sample["prompt"]
        ),
        "target_text": sample["summary"],
    }


# --------------------------------------------------
# Tokenization
# --------------------------------------------------
def tokenize_t5(tokenizer, max_input_len, max_target_len):
    def _tok(batch):
        model_inputs = tokenizer(
            batch["input_text"],
            truncation=True,
            max_length=max_input_len,
        )

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                batch["target_text"],
                truncation=True,
                max_length=max_target_len,
            )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return _tok


# --------------------------------------------------
# Main training
# --------------------------------------------------
def main():
    cfg = load_config()

    model_name = cfg["model"]["name"]

    # Load data
    train_samples = list(build_pts_samples(cfg["data"]["train_path"]))[:50000]
    val_samples   = list(build_pts_samples(cfg["data"]["val_path"]))[:5000]

    train_ds = Dataset.from_list([format_t5(s) for s in train_samples])
    val_ds   = Dataset.from_list([format_t5(s) for s in val_samples])

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    tokenized_train = train_ds.map(
        tokenize_t5(
            tokenizer,
            cfg["model"]["max_input_length"],
            cfg["model"]["max_target_length"],
        ),
        batched=True,
        remove_columns=train_ds.column_names,
    )

    tokenized_val = val_ds.map(
        tokenize_t5(
            tokenizer,
            cfg["model"]["max_input_length"],
            cfg["model"]["max_target_length"],
        ),
        batched=True,
        remove_columns=val_ds.column_names,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
    )

    args = TrainingArguments(
        output_dir=cfg["training"]["output_dir"],
        per_device_train_batch_size=cfg["training"]["train_batch_size"],
        per_device_eval_batch_size=cfg["training"]["eval_batch_size"],
        gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
        num_train_epochs=cfg["training"]["num_train_epochs"],
        learning_rate=cfg["training"]["learning_rate"],
        fp16=cfg["training"]["fp16"],
        evaluation_strategy="steps",
        eval_steps=cfg["training"]["eval_steps"],
        save_steps=cfg["training"]["save_steps"],
        logging_steps=cfg["training"]["logging_steps"],
        save_total_limit=2,
        report_to=[],
        predict_with_generate=True,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(cfg["training"]["output_dir"])
    tokenizer.save_pretrained(cfg["training"]["output_dir"])


if __name__ == "__main__":
    main()
