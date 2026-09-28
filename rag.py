import os
import time
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.chat_history import InMemoryChatMessageHistory
from router import route_query
from dense import dense_retrieve
from hybrid import hybrid_retrieve
from multi_query import multi_query_retrieve
from hyde import hyde_retrieve
from parent_doc import parent_document_retrieve
from Reranker import rerank_documents
from compression import compress_context
from reordering import reorder_documents
load_dotenv()
llm = ChatOpenAI(
    model="nvidia/nemotron-3.5-lightning:free",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Answer strictly only from the provided context.
            If the answer cannot be found in the context,
            say that the information is not available.
            Do not use outside knowledge.
            Context:
            {context}
            Chat History:
            {chat_history}
            """
        ),
        (
            "human",
            "{input}"
        )
    ]
)
PIPELINES = {
    "dense": dense_retrieve,
    "hybrid": hybrid_retrieve,
    "multi_query": multi_query_retrieve,
    "hyde": hyde_retrieve,
    "parent_document": parent_document_retrieve
}
def metadata_filter(docs):
    if not docs:
        return docs
    return docs
def apply_common_enhancements(
    query,
    docs
):
    docs = metadata_filter(
        docs
    )
    print(
        "\nAfter metadata filtering:",
        len(docs)
    )
    if docs:
        docs = rerank_documents(
            query=query,
            docs=docs,
            top_k=5
        )
    print(
        "After reranking:",
        len(docs)
    )
    if docs:
        docs = compress_context(
            query=query,
            docs=docs,
            top_sentences=5
        )
    print(
        "After compression:",
        len(docs)
    )
    if docs:
        docs = reorder_documents(
            docs
        )
    print(
        "After reordering:",
        len(docs)
    )
    return docs
def retrieve_context(query: str):
    router_result = route_query(
        query
    )
    selected_pipeline = (
        router_result["pipeline"]
    )
    router_confidence = (
        router_result["confidence"]
    )
    router_probabilities = (
        router_result["probabilities"]
    )
    print(
        "Query:",
        query
    )
    print(
        "Selected pipeline:",
        selected_pipeline
    )
    print(
        "Router confidence:",
        router_confidence
    )
    if router_probabilities:
        print(
            "Pipeline probabilities:"
        )
        for pipeline, probability in (
            router_probabilities.items()
        ):
            print(
                f"  {pipeline}: "
                f"{probability:.4f}"
            )
    if selected_pipeline not in PIPELINES:
        raise ValueError(
            f"Unknown retrieval pipeline: "
            f"{selected_pipeline}"
        )
    retriever = PIPELINES[
        selected_pipeline
    ]
    retrieval_start = time.perf_counter()
    docs = retriever(
        query
    )
    retrieval_latency = (
        time.perf_counter()
        - retrieval_start
    )

    print(
        "RETRIEVAL RESULTS"
    )

    print(
        "Pipeline:",
        selected_pipeline
    )
    print(
        "Retrieved documents:",
        len(docs)
    )
    print(
        f"Retrieval latency: "
        f"{retrieval_latency:.2f}s"
    )
    enhancement_start = time.perf_counter()
    docs = apply_common_enhancements(
        query=query,
        docs=docs
    )
    enhancement_latency = (
        time.perf_counter()
        - enhancement_start
    )
    print(
        f"\nEnhancement latency: "
        f"{enhancement_latency:.2f}s"
    )
    return {
        "docs":
            docs,
        "pipeline":
            selected_pipeline,
        "router_confidence":
            router_confidence,
        "router_probabilities":
            router_probabilities,
        "retrieval_latency":
            retrieval_latency,
        "enhancement_latency":
            enhancement_latency
    }
def format_docs(docs):
    if not docs:
        return ""
    return "\n\n".join(
        doc.page_content
        for doc in docs
    )
def generate_answer_and_context(
    query: str
):

    total_start = time.perf_counter()
    retrieval_result = retrieve_context(
        query
    )
    docs = retrieval_result["docs"]
    pipeline = retrieval_result["pipeline"]
    router_confidence = (
        retrieval_result["router_confidence"]
    )
    router_probabilities = (
        retrieval_result["router_probabilities"]
    )
    retrieval_latency = (
        retrieval_result["retrieval_latency"]
    )
    enhancement_latency = (
        retrieval_result["enhancement_latency"]
    )
    context = format_docs(
        docs
    )
    generation_start = time.perf_counter()
    messages = prompt.format_messages(
        context=context,
        chat_history="",
        input=query
    )
    response = llm.invoke(
        messages
    )
    generation_latency = (
        time.perf_counter()
        - generation_start
    )
    total_latency = (
        time.perf_counter()
        - total_start
    )

    print(
        "LATENCY"
    )

    print(
        f"Retrieval latency: "
        f"{retrieval_latency:.2f}s"
    )
    print(
        f"Enhancement latency: "
        f"{enhancement_latency:.2f}s"
    )
    print(
        f"Generation latency: "
        f"{generation_latency:.2f}s"
    )
    print(
        f"Total RAG latency: "
        f"{total_latency:.2f}s"
    )
    return {
        "question":
            query,
        "answer":
            response.content,
        "pipeline":
            pipeline,
        "router_confidence":
            router_confidence,
        "router_probabilities":
            router_probabilities,
        "contexts": [
            doc.page_content
            for doc in docs
        ],
        "latency":
            total_latency,
        "retrieval_latency":
            retrieval_latency,
        "enhancement_latency":
            enhancement_latency,
        "generation_latency":
            generation_latency
    }
store = {}
def get_session_history(
    session_id: str
):

    if session_id not in store:
        store[session_id] = (
            InMemoryChatMessageHistory()
        )
    return store[
        session_id
    ]
def ask_question(
    session_id: str,
    query: str
):

    result = generate_answer_and_context(
        query
    )
    return result
if __name__ == "__main__":
    session_id = "test-session"
    while True:
        query = input(
            "\nAsk Question: "
        )
        if query.lower().strip() == "exit":
            break
        try:
            result = (
                generate_answer_and_context(
                    query
                )
            )
            print(
                "\n" + "=" * 70
            )
            print(
                "ANSWER"
            )
            print(
                "=" * 70
            )
            print(
                result["answer"]
            )
            print(
                "\nPipeline:",
                result["pipeline"]
            )
            print(
                "Router confidence:",
                result["router_confidence"]
            )
            print(
                "Total latency:",
                f"{result['latency']:.2f}s"
            )
        except Exception as e:
            print(
                "\nError:",
                str(e)
            )
