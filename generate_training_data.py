import pandas as pd
import time
import json
from query_analyzer import analyze_query
from dense import dense_retrieve
from hybrid import hybrid_retrieve
from multi_query import multi_query_retrieve
from hyde import hyde_retrieve
from parent_doc import parent_document_retrieve
from evaluate_pipelines import (
    evaluate_answer,
    calculate_score
)
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
load_dotenv()
llm = ChatOpenAI(
    model="nvidia/nemotron-3.5-lightning:free",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)
PIPELINES = {
    "dense": dense_retrieve,
    "hybrid": hybrid_retrieve,
    "multi_query": multi_query_retrieve,
    "hyde": hyde_retrieve,
    "parent_document": parent_document_retrieve
}
def generate_answer(query, docs):
    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
Answer only using the provided context.
If the answer cannot be found in the context,
say that the information is not available.
Context:
{context}
"""
            ),
            ("human", "{query}")
        ]
    )
    messages = prompt.format_messages(
        context=context,
        query=query
    )
    response = llm.invoke(messages)
    return response.content
def run_pipeline(query, pipeline_name):
    retriever = PIPELINES[pipeline_name]
    start_time = time.perf_counter()
    docs = retriever(query)
    retrieval_time = (
        time.perf_counter() - start_time
    )
    answer_start = time.perf_counter()
    answer = generate_answer(
        query,
        docs
    )
    answer_time = (
        time.perf_counter() - answer_start
    )
    contexts = [
        doc.page_content
        for doc in docs
    ]
    evaluation = evaluate_answer(
        query,
        answer,
        contexts
    )
    score = calculate_score(
        evaluation
    )
    return {
        "pipeline": pipeline_name,
        "answer": answer,
        "contexts": contexts,
        "retrieval_count": len(docs),
        "retrieval_latency": retrieval_time,
        "answer_latency": answer_time,
        "total_latency":
            retrieval_time + answer_time,
        "relevance":
            evaluation["relevance"],
        "faithfulness":
            evaluation["faithfulness"],
        "completeness":
            evaluation["completeness"],
        "score": score
    }
def main():
    input_file = "seed_queries.csv"
    output_file = "router_dataset.csv"
    df = pd.read_csv(input_file)
    all_records = []
    for index, row in df.iterrows():
        query = row["query"]
        print("\n")
        print(f"QUERY {index + 1}/{len(df)}")
        print(query)
        features = analyze_query(query)
        print("\nQuery Features:")
        print(features)
        pipeline_results = []
        for pipeline_name in PIPELINES:
            print(
                f"\nRunning pipeline: "
                f"{pipeline_name}"
            )
            try:
                result = run_pipeline(
                    query,
                    pipeline_name
                )
                pipeline_results.append(
                    result
                )
                print(
                    f"Score: "
                    f"{result['score']:.4f}"
                )
            except Exception as e:
                print(
                    f"Pipeline failed: "
                    f"{pipeline_name}"
                )
                print(e)
        if not pipeline_results:
            print("All pipelines failed.")
            continue
        best_result = max(
            pipeline_results,
            key=lambda x: x["score"]
        )
        print("\nBEST PIPELINE:")
        print(best_result["pipeline"])
        print(
            "BEST SCORE:",
            best_result["score"]
        )
        record = {
            "query": query,
            "query_type":
                features.get(
                    "query_type"
                ),
            "complexity":
                features.get(
                    "complexity"
                ),
            "multi_concept":
                features.get(
                    "multi_concept"
                ),
            "keyword_heavy":
                features.get(
                    "keyword_heavy"
                ),
            "requires_multiple_documents":
                features.get(
                    "requires_multiple_documents"
                ),
            "requires_exact_match":
                features.get(
                    "requires_exact_match"
                ),
            "best_pipeline":
                best_result["pipeline"],
            "best_score":
                best_result["score"],
            "best_latency":
                best_result["total_latency"],
            "all_results":
                json.dumps(
                    pipeline_results
                )
        }
        all_records.append(record)
    output_df = pd.DataFrame(
        all_records
    )
    output_df.to_csv(
        output_file,
        index=False
    )
    print("\n")
    print(
        f"Saved to: {output_file}"
    )
    print(
        "\nPipeline distribution:"
    )
    print(
        output_df[
            "best_pipeline"
        ].value_counts()
    )
if __name__ == "__main__":
    main()
