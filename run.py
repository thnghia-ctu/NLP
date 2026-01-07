import json
from src.utils.data_utils import read_jsonl

def main():
    path = "data/processed/train_subset.jsonl"
    path1 = "data/processed/output.txt"

    # print(next(read_jsonl(path))['abstract'])

    item = next(read_jsonl(path1))
    print(item.keys())
    print(item['abstract_text'])

if __name__ == "__main__":
    main()
