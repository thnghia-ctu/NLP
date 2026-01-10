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
        sent = clean_article(sent).strip()

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


def clean_article(latex_text):
    """
    Làm sạch văn bản LaTeX một cách toàn diện, loại bỏ tất cả các lệnh và ký tự đặc biệt.

    Args:
        latex_text (str): Văn bản LaTeX cần làm sạch

    Returns:
        str: Văn bản đã được làm sạch
    """
    if not latex_text:
        return ""

    text = latex_text

    # Loại bỏ preamble (phần trước \begin{document})
    text = re.sub(r"\\documentclass.*?\\begin\{document\}", "", text, flags=re.DOTALL)
    text = re.sub(r"\\end\{document\}.*", "", text, flags=re.DOTALL)

    # Loại bỏ comments (% và nội dung sau nó)
    text = re.sub(r"%.*?(\n|$)", "\n", text)

    # Loại bỏ các môi trường toán học nhiều dòng
    math_envs = [
        "equation",
        "align",
        "gather",
        "multline",
        "flalign",
        "alignat",
        "eqnarray",
        "displaymath",
        "math",
        "array",
    ]
    for env in math_envs:
        text = re.sub(
            rf"\\begin\{{{env}\*?\}}.*?\\end\{{{env}\*?\}}", " ", text, flags=re.DOTALL
        )

    # Loại bỏ các môi trường figure, table
    text = re.sub(
        r"\\begin\{figure\*?\}.*?\\end\{figure\*?\}", " ", text, flags=re.DOTALL
    )
    text = re.sub(
        r"\\begin\{table\*?\}.*?\\end\{table\*?\}", " ", text, flags=re.DOTALL
    )
    text = re.sub(r"\\begin\{tabular\}.*?\\end\{tabular\}", " ", text, flags=re.DOTALL)

    # Loại bỏ các môi trường khác (verbatim, lstlisting, tikzpicture, etc.)
    other_envs = [
        "verbatim",
        "lstlisting",
        "code",
        "tikzpicture",
        "algorithm",
        "algorithmic",
        "proof",
        "abstract",
    ]
    for env in other_envs:
        text = re.sub(
            rf"\\begin\{{{env}\*?\}}.*?\\end\{{{env}\*?\}}", " ", text, flags=re.DOTALL
        )

    # Loại bỏ inline math: $...$, \(...\)
    text = re.sub(r"\$\$.*?\$\$", " ", text, flags=re.DOTALL)
    text = re.sub(r"\$[^\$]*?\$", " ", text)
    text = re.sub(r"\\\(.*?\\\)", " ", text, flags=re.DOTALL)
    text = re.sub(r"\\\[.*?\\\]", " ", text, flags=re.DOTALL)

    # Xử lý các lệnh phổ biến có nội dung cần giữ lại
    # \text{...}, \emph{...}, \textit{...}, \textbf{...}, etc.
    text_commands = [
        "text",
        "emph",
        "textit",
        "textbf",
        "texttt",
        "textrm",
        "textsf",
        "textsc",
        "textsl",
        "textup",
        "textmd",
        "mathrm",
        "mathbf",
        "mathit",
        "mathsf",
        "mathtt",
        "mathcal",
        "mathbb",
    ]
    for cmd in text_commands:
        text = re.sub(rf"\\{cmd}\{{([^}}]*)\}}", r"\1", text)

    # Loại bỏ các lệnh citation và reference
    text = re.sub(r"\\cite\{[^}]*\}", " ", text)
    text = re.sub(r"\\ref\{[^}]*\}", " ", text)
    text = re.sub(r"\\label\{[^}]*\}", " ", text)
    text = re.sub(r"\\eqref\{[^}]*\}", " ", text)
    text = re.sub(r"\\pageref\{[^}]*\}", " ", text)

    # Loại bỏ các lệnh sectioning nhưng giữ lại tiêu đề
    section_commands = [
        "part",
        "chapter",
        "section",
        "subsection",
        "subsubsection",
        "paragraph",
        "subparagraph",
    ]
    for cmd in section_commands:
        text = re.sub(rf"\\{cmd}\*?\{{([^}}]*)\}}", r"\1. ", text)

    # Loại bỏ footnote, margin notes
    text = re.sub(r"\\footnote\{[^}]*\}", " ", text)
    text = re.sub(r"\\marginpar\{[^}]*\}", " ", text)

    # Loại bỏ các lệnh graphics
    text = re.sub(r"\\includegraphics.*?\{[^}]*\}", " ", text)
    text = re.sub(r"\\caption\{[^}]*\}", " ", text)

    # Loại bỏ bibliography
    text = re.sub(r"\\bibliography\{[^}]*\}", " ", text)
    text = re.sub(r"\\bibliographystyle\{[^}]*\}", " ", text)

    # Loại bỏ các lệnh URL và hyperlink
    text = re.sub(r"\\url\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\href\{[^}]*\}\{([^}]*)\}", r"\1", text)

    # Loại bỏ các môi trường list (itemize, enumerate, description)
    text = re.sub(r"\\item\s*\[.*?\]", "", text)
    text = re.sub(r"\\item", "", text)
    text = re.sub(r"\\begin\{(itemize|enumerate|description)\*?\}", "", text)
    text = re.sub(r"\\end\{(itemize|enumerate|description)\*?\}", "", text)

    # Loại bỏ tất cả các lệnh còn lại có dạng \command{...}
    # Thực hiện nhiều lần để xử lý nested commands
    for _ in range(5):
        text = re.sub(r"\\[a-zA-Z]+\*?\{[^{}]*\}", " ", text)

    # Loại bỏ các lệnh optional parameters [...]
    text = re.sub(r"\\[a-zA-Z]+\*?\[[^\]]*\]", " ", text)

    # Loại bỏ các lệnh không có tham số
    text = re.sub(r"\\[a-zA-Z]+\*?", " ", text)

    # Xử lý các ký tự đặc biệt của LaTeX
    replacements = {
        r"``": '"',
        r"''": '"',
        r"`": "'",
        r"\\&": "&",
        r"\\%": "%",
        r"\\_": "_",
        r"\\#": "#",
        r"\\$": "$",
        r"---": "—",
        r"--": "–",
        r"~": " ",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    # Loại bỏ các ký tự đặc biệt còn lại
    text = re.sub(r"[{}\\]", " ", text)
    text = re.sub(r"[\[\]]", " ", text)
    text = re.sub(r"[&^_]", " ", text)

    # Chuẩn hóa khoảng trắng
    text = re.sub(r"\n\s*\n", "\n\n", text)  # Giữ ngắt đoạn
    text = re.sub(r" +", " ", text)  # Nhiều space thành 1
    text = re.sub(r"\n ", "\n", text)  # Bỏ space đầu dòng
    text = re.sub(r" \n", "\n", text)  # Bỏ space cuối dòng

    # Loại bỏ citation dạng @xcite, @cite, @ref, @xref
    text = re.sub(r"@\s*(xcite|cite|ref|xref)\b", " ", text, flags=re.IGNORECASE)
    # Loại bỏ token math artifact như @xmath10, @xmath3
    text = re.sub(r"@\s*xmath\d+\b", " ", text, flags=re.IGNORECASE)
    # Loại bỏ tất cả sentence tags <S>, </S>
    text = re.sub(r"</?S>", " ", text, flags=re.IGNORECASE)

    # Loại bỏ khoảng trắng đầu/cuối
    text = text.strip()

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
