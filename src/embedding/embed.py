import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def embed_batch(texts, model: str = "text-embedding-3-large"):
    """
    Embed a list of texts in one API call.
    Returns a list of embedding vectors, one per input text.
    """
    response = client.embeddings.create(
        model=model,
        input=texts
    )
    return [item.embedding for item in response.data]

def embed_text(text: str, model: str = "text-embedding-3-large"):
    """
    Backwards-compatible helper for embedding a single text.
    Internally uses embed_batch([text]) to keep behavior consistent.
    """
    return embed_batch([text], model=model)[0]