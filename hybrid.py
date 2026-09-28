from langchain_classic.retrievers import EnsembleRetriever
from dense import dense_retriever
from retriever import create_bm25
bm25_retriever = create_bm25()
hybrid_retriever = EnsembleRetriever(
    retrievers=[
        dense_retriever,
        bm25_retriever
    ],
    weights=[
        0.4,
        0.6
    ]
)
def hybrid_retrieve(query: str):

    return hybrid_retriever.invoke(query)
