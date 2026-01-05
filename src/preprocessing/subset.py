from datasets import load_dataset
import json
import yaml
import os

def create_subset(config_path="configs/config_data.yaml"):
    # Load config
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    dataset_name = cfg["dataset"]["name"]   # "ccdv/arxiv-summarization" hoặc "ccdv/pubmed-summarization"
    subset_size = cfg["dataset"]["subset_size"]

    # Load dataset
    dataset = load_dataset(dataset_name)

    # Tạo subset
    small_train = dataset["train"].shuffle(seed=42).select(range(subset_size))
    small_val   = dataset["validation"].shuffle(seed=42).select(range(int(subset_size*0.1)))
    small_test  = dataset["test"].shuffle(seed=42).select(range(int(subset_size*0.1)))

    # Lưu ra JSONL
    os.makedirs("data/processed", exist_ok=True)
    def save_jsonl(split, filename):
        with open(filename, "w", encoding="utf-8") as f:
            for sample in split:
                f.write(json.dumps(sample) + "\n")

    save_jsonl(small_train, "data/processed/train_subset.jsonl")
    save_jsonl(small_val, "data/processed/val_subset.jsonl")
    save_jsonl(small_test, "data/processed/test_subset.jsonl")

if __name__ == "__main__":
    create_subset()
