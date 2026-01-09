import json
from src.utils.data_utils import read_jsonl
from src.models.mistral_baseline import summarize_section
from src.preprocessing.data import ArticleSample, parse_item, json_to_sample, build_pts_samples
from rouge_score import rouge_scorer
from src.run_pts import merge_section_summaries

def main():

    path = "data/processed/output.txt"

    # 1. Đọc article
    articles = [parse_item(x) for x in read_jsonl(path)]

    # 2. Lấy article đầu tiên (test trước cho an toàn)
    first_article = articles[0]

    # 3. Build PTS samples (list[dict])
    pts_dicts = build_pts_samples(first_article)

    # 4. Chuyển sang PTSSample
    pts_samples = [json_to_sample(d) for d in pts_dicts]

    for pts in pts_samples:
        if(len(pts.target)>0 ):
            print(pts.source)
            print('---------------------------------------------------------------------')
            print('---------------------------------------------------------------------')
            print(pts.target)
            break

    # 5. Tóm tắt từng section
    pred_sections = []
    for s in pts_samples:
        sec_summary = summarize_section(s.source)
        pred_sections.append((s.section_idx, sec_summary))

    # 6. Gộp thành abstract toàn cục
    first_article.pred_summary = merge_section_summaries(pred_sections)

if __name__ == "__main__":
    main() 
