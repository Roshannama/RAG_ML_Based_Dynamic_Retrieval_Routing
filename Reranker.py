from sentence_transformers import CrossEncoder
reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)
def rerank_documents(query, docs, top_k=5):
    pairs = [
        (query, doc.page_content)
        for doc in docs
    ]
    scores = reranker.predict(pairs)
    ranked_docs = sorted(
        zip(docs, scores),
        key=lambda x: x[1],
        reverse=True
    )
    return [
        doc
        for doc, score in ranked_docs[:top_k]
    ]
