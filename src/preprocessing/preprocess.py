import re

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
