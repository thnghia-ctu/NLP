from dataclasses import dataclass
from src.preprocessing.preprocess import split_sentences, align_abstract_to_sections, chunk_sentences, clean_article

@dataclass
class ArticleSample:
    id: str
    text: str
    summary: list
    labels: list
    sections: list
    section_names: list
    pred_summary: str

def parse_item(item):
    return ArticleSample(
        id=item["article_id"],
        text=item["article_text"],
        summary=item["abstract_text"],
        labels=item["labels"],
        sections=item["sections"],
        section_names=item["section_names"],
        pred_summary=''
    )

@dataclass
class PTSSample:
    article_id: str
    section_name: str
    section_idx: int
    source: str
    target: list

def json_to_sample(j):
    return PTSSample(
        article_id=j["article_id"],
        section_name=j["section_name"],
        section_idx=j["section_idx"],
        source=j["section_text"],
        target=j["abstract_section"]
    )

def build_pts_samples(article: ArticleSample):
    sections = article.sections 
    names = article.section_names
    abstract = clean_article("".join(article.summary))                # List[str]

    # abstract -> list câu
    abstract_sents = split_sentences(abstract)
    
    samples = []
    idx=0
    for i, sec in enumerate(sections):
        chunks=chunk_sentences(sec)
        for p in chunks:
            samples.append({
                "article_id": article.id,
                "section_name": names[i],
                "section_text": p,
                "abstract_section": [],
                "section_idx": idx
            })
            idx=idx+1
    return align_abstract_to_sections(samples, abstract_sents)
