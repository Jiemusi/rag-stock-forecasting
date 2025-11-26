import textwrap

MAX_CHARS = 3000

def chunk_text(text: str, max_chars: int = MAX_CHARS):
    """
    Split the text into manageable chunks for embedding.
    Uses soft boundaries (split at sentence/paragraph).
    """
    if len(text) <= max_chars:
        return [text]

    chunks = []
    paragraphs = text.split("\n")

    buffer = ""

    for para in paragraphs:
        if len(buffer) + len(para) + 1 <= max_chars:
            buffer += para + "\n"
        else:
            chunks.append(buffer.strip())
            buffer = para + "\n"

    if buffer.strip():
        chunks.append(buffer.strip())

    return chunks