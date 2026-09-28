from langchain_community.retrievers import BM25Retriever
from langchain_community.document_loaders import DirectoryLoader,PyPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
def create_bm25():
    md_loader = DirectoryLoader(
        "source_data",
        glob="**/*.md",
        loader_cls=TextLoader
    )
    pdf_loader = DirectoryLoader(
        "source_data",
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = md_loader.load() + pdf_loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    docs = splitter.split_documents(documents)
    bm25 = BM25Retriever.from_documents(docs)
    bm25.k = 20
    return bm25
