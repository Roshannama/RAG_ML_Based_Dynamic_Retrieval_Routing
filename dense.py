import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
load_dotenv()
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
EMBED_MODEL = os.getenv("EMBED_MODEL")
embedding = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL
)
vectorstore = PineconeVectorStore(
    index_name=INDEX_NAME,
    embedding=embedding
)
dense_retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 20}
)
def dense_retrieve(query: str):
    return dense_retriever.invoke(query)
