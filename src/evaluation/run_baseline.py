from src.utils.data_utils import read_jsonl
from src.models.mistral_baseline import summarize_article
from src.preprocessing.data import ArticleSample, parse_item
from rouge_score import rouge_scorer

articles = [parse_item(x) for x in read_jsonl("data/processed/output.txt")]
scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
# print(articles[0].summary[0])

for i, art in enumerate(articles):
    art.pred_summary = summarize_article(art.text)
    scores = scorer.score(art.summary[0], art.pred_summary)

    print(f"\nArticle {i} | ID: {art.id}")
    print(f"ROUGE-1: {scores['rouge1'].fmeasure:.4f}")
    print(f"ROUGE-2: {scores['rouge2'].fmeasure:.4f}")
    print(f"ROUGE-L: {scores['rougeL'].fmeasure:.4f}")

    if i == 4:
        break
