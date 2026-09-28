import os
import json
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
load_dotenv()
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
EMBED_MODEL = os.getenv("EMBED_MODEL")
PARENT_DIR = "data/parents"
embedding = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL
)
vectorstore = PineconeVectorStore(
    index_name=INDEX_NAME,
    embedding=embedding
)
child_retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 10
    }
)
def load_parent(parent_id):
    path = os.path.join(
        PARENT_DIR,
        f"{parent_id}.json"
    )
    if not os.path.exists(path):
        return None
    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:
        data = json.load(f)
    return data
def parent_document_retrieve(
    query,
    k=5
):
    child_docs = child_retriever.invoke(query)
    parent_ids = []
    for doc in child_docs:
        parent_id = doc.metadata.get(
            "parent_id"
        )
        if parent_id and parent_id not in parent_ids:
            parent_ids.append(parent_id)
    parent_docs = []
    for parent_id in parent_ids[:k]:
        parent = load_parent(parent_id)
        if parent is None:
            continue
        parent_docs.append(parent)
    return parent_docs
