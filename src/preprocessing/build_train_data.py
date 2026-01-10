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

            raw_article = obj["article"]
            raw_abstract = obj["abstract"]

            sections = segment_article(raw_article)
            abs_sents = split_sentences(raw_abstract)

            # aligned = list[ {section_text, abstract_section} ]
            aligned = align_abstract_to_sections(sections, abs_sents)

            for sec in aligned:
                raw_sec_text = sec.get("section_text", "")
                raw_sum_sents = sec.get("abstract_section", [])

                # Clean SAU khi align
                cleaned_sec = clean_article(raw_sec_text)
                cleaned_summary = clean_article(" ".join(raw_sum_sents))

                if len(cleaned_sec.strip()) < 200:
                    continue
                if not cleaned_summary.strip():
                    continue

                yield {
                    "prompt": f"Summarize the following section:\n{cleaned_sec}\nSummary:",
                    "summary": cleaned_summary,
                }