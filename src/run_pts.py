from collections import defaultdict
from src.models.mistral_baseline import summarize_section
from src.preprocessing.data import PTSSample

def run_pts_inference(pts_samples: list[PTSSample]):
    article_summaries = defaultdict(list)

    for s in pts_samples:
        sec_summary = summarize_section(s.source)
        article_summaries[s.article_id].append(
            (s.section_idx, sec_summary)
        )

    return article_summaries

def merge_section_summaries(section_summaries):
    section_summaries = sorted(section_summaries, key=lambda x: x[0])
    return " ".join([s for _, s in section_summaries])

def summarize_global(section_summaries):
    text = merge_section_summaries(section_summaries)
    return summarize_section(text, max_new_tokens=200)
