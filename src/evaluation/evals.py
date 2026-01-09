from rouge_score import rouge_scorer
from bert_score import score

def rouge_eval(articles):
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )
    for art in articles:
        ref = " ".join(art.summary)
        pred = art.pred_summary

        scores = scorer.score(ref, pred)

        print(f"\nArticle {art.id}")
        print(f"R1: {scores['rouge1'].fmeasure:.4f}")
        print(f"R2: {scores['rouge2'].fmeasure:.4f}")
        print(f"RL: {scores['rougeL'].fmeasure:.4f}")



def bertscore_eval(articles, lang="en", model_type="microsoft/deberta-xlarge-mnli"):
    """
    BERTScore evaluation for abstractive summarization.
    """

    refs = []
    preds = []
    ids = []

    for art in articles:
        refs.append(" ".join(art.summary))   # reference abstract
        preds.append(art.pred_summary)        # generated abstract
        ids.append(art.id)

    # Compute BERTScore
    P, R, F1 = score(
        preds,
        refs,
        lang=lang,
        model_type=model_type,
        verbose=True
    )

    # Print per-article score
    for i, art_id in enumerate(ids):
        print(f"\nArticle {art_id}")
        print(f"BERTScore-P:  {P[i].item():.4f}")
        print(f"BERTScore-R:  {R[i].item():.4f}")
        print(f"BERTScore-F1: {F1[i].item():.4f}")

    # Print average
    print("\n===== Average BERTScore =====")
    print(f"P:  {P.mean().item():.4f}")
    print(f"R:  {R.mean().item():.4f}")
    print(f"F1: {F1.mean().item():.4f}")
