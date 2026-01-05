import json
from src.utils.data_utils import read_jsonl

def main():
    path = "data/processed/train_subset.jsonl"

    print(next(read_jsonl(path))['abstract'])

    # for i, sample in enumerate(read_jsonl(path)):
    #     print(f"ID: {sample['id']}")
    #     print(f"Abstract: {sample['abstract']}")
    #     print(f"Article (100 ký tự đầu): {sample['article'][:100]}")
    #     print("-" * 50)

    #     if i == 1:  # chỉ in thử 3 mẫu đầu
    #         break

if __name__ == "__main__":
    main()
