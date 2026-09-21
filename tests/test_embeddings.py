"""Smoke test: embeddings are fixed-length vectors, and cosine similarity
is higher for related sentences than unrelated ones. That similarity
ranking is the entire mechanism semantic search relies on.

Requires a running local Ollama server with the nomic-embed-text model pulled.
"""

import numpy as np

from embeddings.provider import get_embeddings


def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def test_embeddings_are_semantic():
    embeddings = get_embeddings()

    v_dog = embeddings.embed_query("The dog played fetch in the park.")
    v_puppy = embeddings.embed_query("A puppy ran around chasing a ball outside.")
    v_finance = embeddings.embed_query("The quarterly earnings report exceeded expectations.")

    assert len(v_dog) == len(v_puppy) == len(v_finance)
    assert len(v_dog) > 0

    sim_related = cosine_similarity(v_dog, v_puppy)
    sim_unrelated = cosine_similarity(v_dog, v_finance)

    assert sim_related > sim_unrelated, (
        f"expected related sentences to be more similar: "
        f"related={sim_related:.3f} unrelated={sim_unrelated:.3f}"
    )


if __name__ == "__main__":
    test_embeddings_are_semantic()
    print("embeddings: OK")
