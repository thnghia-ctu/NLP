import json
from src.preprocessing.preprocess import (
    clean_article,
    segment_article,
    split_sentences,
    align_abstract_to_sections,
)


def build_pts_samples(jsonl_path):
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)

            article = clean_article(obj["article"])
            abstract = clean_article(obj["abstract"])

            sections = segment_article(article)
            abs_sents = split_sentences(abstract)
            aligned = align_abstract_to_sections(sections, abs_sents)

            for i, sec in enumerate(sections):
                if len(sec.strip()) < 200:
                    continue

                yield {
                    "prompt": f"Summarize the following section:\n{sec}\nSummary:",
                    "summary": aligned[i],
                }
