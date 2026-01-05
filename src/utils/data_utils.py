# src/utils/data_utils.py
import json
from typing import List, Dict
import random

def read_jsonl(path: str):
  with open(path, "r", encoding="utf-8") as f:
    for line in f:
      yield json.loads(line)

def join_sentences(sent_list: List[str], max_n: int):
  return " ".join(sent_list[:max_n])

def build_samples(path: str, input_field: str, target_field: str,
                  prompt_template: str,
                  max_input_sentences: int, max_target_sentences: int):
  for ex in read_jsonl(path):
    # Lấy input từ article_text hoặc sections
    if input_field == "article_text":
      article = join_sentences(ex["article_text"], max_input_sentences)
    elif input_field == "sections":
      # flatten sections thành câu
      flat = [s for sec in ex["sections"] for s in sec]
      article = join_sentences(flat, max_input_sentences)
    else:
      raise ValueError("Unknown input_field")

    # Target từ abstract_text
    target = join_sentences(ex["abstract_text"], max_target_sentences)

    prompt = prompt_template.format(article=article)
    yield {"prompt": prompt, "summary": target}

def split_for_debug(samples, n=None):
  # lấy n mẫu đầu để debug
  data = list(samples)
  return data[:n] if n else data
