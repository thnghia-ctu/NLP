import re
import nltk
from sentence_transformers import SentenceTransformer, util
import torch

MODEL = SentenceTransformer("all-MiniLM-L6-v2")

def split_sentences(text):
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")
        nltk.download("punkt_tab")
    return nltk.sent_tokenize(" ".join(text) if isinstance(text, list) else text)

def align_abstract_to_sections(sections, abstract_sents):
    sim = compute_similarity(sections, abstract_sents)
    for i, sent in enumerate(abstract_sents):
        best_sec = torch.argmax(sim[i]).item()

        sections[best_sec].setdefault("abstract_section", [])
        sections[best_sec]["abstract_section"].append(sent)
    return sections

def compute_similarity(sections: list[dict], abstract_sents: list[str], model=MODEL):
    """
    Returns:
        sim_matrix: Tensor shape (num_abstract_sents, num_sections)
    """

    # 1. Load model (load 1 lần, có thể đưa ra ngoài)
    # model = SentenceTransformer("all-MiniLM-L6-v2")

    # 2. Lấy text
    section_texts = [s["section_text"] for s in sections]

    # 3. Encode
    abs_emb = model.encode(
        abstract_sents,
        convert_to_tensor=True,
        normalize_embeddings=True
    )  # (A, D)

    sec_emb = model.encode(
        section_texts,
        convert_to_tensor=True,
        normalize_embeddings=True
    )  # (S, D)

    # 4. Cosine similarity
    sim_matrix = util.cos_sim(abs_emb, sec_emb)  
    # shape: (num_abstract_sents, num_sections)

    return sim_matrix

def chunk_sentences(sentences, max_chars=1024):
    """
    sentences: list[str]
    return: list[str]  # mỗi phần <= max_chars
    """

    chunks = []
    current = ""

    for sent in sentences:
        sent = sent.strip()

        # Nếu 1 câu đã quá dài → cắt cưỡng bức (hiếm)
        if len(sent) > max_chars:
            if current:
                chunks.append(current.strip())
                current = ""
            chunks.append(sent[:max_chars])
            continue

        if len(current) + len(sent) <= max_chars:
            current += " " + sent
        else:
            chunks.append(current.strip())
            current = sent

    if current:
        chunks.append(current.strip())

    return chunks


def clean_article(text: str) -> str:
    """Làm sạch LaTeX và citation trong article."""
    text = text.replace("\\n", "\n")
    text = re.sub(r"@xmath\d+", "", text)
    text = re.sub(r"@xcite", "", text)
    return text

def segment_article(article: str, max_chars: int = 2000):
    """Chia article thành các page nhỏ."""
    text = clean_article(article)
    paragraphs = text.split("\n\n")

    pages, current = [], ""
    for para in paragraphs:
        if len(current) + len(para) < max_chars:
            current += para + "\n\n"
        else:
            pages.append(current.strip())
            current = para + "\n\n"
    if current:
        pages.append(current.strip())

    return pages
