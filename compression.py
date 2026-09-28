import os
import re
from langchain_core.documents import Document
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
load_dotenv()
embedding_model = HuggingFaceEmbeddings(
    model_name=os.getenv("EMBED_MODEL")
)
def split_sentences(text):
    sentences = re.split(
        r'(?<=[.!?])\s+',
        text
    )
    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]
def compress_document(
    query,
    document,
    top_sentences=5
):
    sentences = split_sentences(
        document.page_content
    )
    if len(sentences) <= top_sentences:
        return document
    query_embedding = embedding_model.embed_query(
        query
    )
    sentence_embeddings = (
        embedding_model.embed_documents(
            sentences
        )
    )
    import numpy as np
    query_vector = np.array(query_embedding)
    sentence_vectors = np.array(sentence_embeddings)
    query_norm = np.linalg.norm(
        query_vector
    )
    sentence_norms = np.linalg.norm(
        sentence_vectors,
        axis=1
    )
    similarities = (
        sentence_vectors @ query_vector
    ) / (
        sentence_norms * query_norm + 1e-8
    )
    top_indices = np.argsort(
        similarities
    )[-top_sentences:]
    top_indices = sorted(
        top_indices
    )
    selected_sentences = [
        sentences[i]
        for i in top_indices
    ]
    from langchain_core.documents import Document
    return Document(
        page_content="\n".join(
            selected_sentences
        ),
        metadata=document.metadata
    )
def compress_context(
    query,
    docs,
    top_sentences=5
):
    compressed_docs = []
    for doc in docs:
        compressed_doc = compress_document(
            query=query,
            document=doc,
            top_sentences=top_sentences
        )
        compressed_docs.append(
            compressed_doc
        )
    return compressed_docs
