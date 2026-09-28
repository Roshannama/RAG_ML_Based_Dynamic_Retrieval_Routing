import os
import json
import uuid
from dotenv import load_dotenv
from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    PyPDFLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
load_dotenv()
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
EMBED_MODEL = os.getenv("EMBED_MODEL")
PARENT_DIR = "data/parents"
embedding = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL
)
md_loader = DirectoryLoader(
    "source_data",
    glob="**/*.md",
    loader_cls=TextLoader
)
md_docs = md_loader.load()
pdf_loader = DirectoryLoader(
    "source_data",
    glob="**/*.pdf",
    loader_cls=PyPDFLoader
)
pdf_docs = pdf_loader.load()
documents = md_docs + pdf_docs
print(f"Loaded {len(documents)} documents")
parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=200
)
child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=80
)
child_documents = []
parent_count = 0
child_count = 0
for document in documents:
    parent_docs = parent_splitter.split_documents(
        [document]
    )
    for parent_doc in parent_docs:
        parent_id = str(uuid.uuid4())
        parent_count += 1
        parent_data = {
            "parent_id": parent_id,
            "content": parent_doc.page_content,
            "metadata": parent_doc.metadata
        }
        parent_path = os.path.join(
            PARENT_DIR,
            f"{parent_id}.json"
        )
        with open(
            parent_path,
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(
                parent_data,
                f,
                ensure_ascii=False,
                indent=2
            )
        children = child_splitter.split_documents(
            [parent_doc]
        )
        for child_doc in children:
            child_id = str(uuid.uuid4())
            child_count += 1
            child_doc.metadata.update({
                "parent_id": parent_id,
                "child_id": child_id,
                "source": document.metadata.get(
                    "source",
                    "unknown"
                )
            })
            child_documents.append(child_doc)
print(
    f"Created {parent_count} parent documents"
)
print(
    f"Created {child_count} child documents"
)
PineconeVectorStore.from_documents(
    child_documents,
    embedding,
    index_name=INDEX_NAME
)
print("Child documents uploaded to Pinecone successfully.")
