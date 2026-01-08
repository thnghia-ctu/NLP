from dataclasses import dataclass

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
    target: str

def json_to_sample(j):
    return PTSSample(
        article_id=j["article_id"],
        section_name=j["section_name"],
        section_idx=j["section_idx"],
        source=j["section_text"],
        target=j["abstract_section"]
    )

def build_pts_samples(item):
    sections = item["sections"]
    names = item["section_names"]
    abstract = item["abstract_text"]

    abstract_sents = split_sentences(abstract)

    aligned = align_abstract_to_sections(sections, abstract_sents)

    samples = []
    for i, sec in enumerate(sections):
        samples.append({
            "article_id": item["article_id"],
            "section_name": names[i],
            "section_text": sec,
            "abstract_section": aligned[i],
            "section_idx": i
        })
    return samples
