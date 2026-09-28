import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dense import dense_retriever
load_dotenv()
llm = ChatOpenAI(
    model="nvidia/nemotron-3.5-lightning:free",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)
prompt = ChatPromptTemplate.from_template(
    """
    Generate 3 different search queries for the following user question,containing more detail about that query,
    if possible use  full forms or other similar terms to describe the query.
    User question:
    {query}
    Return only the queries.
    """
)
def generate_queries(query: str):
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({
        "query": query
    })
    queries = [
        q.strip()
        for q in response.split("\n")
        if q.strip()
    ]
    return queries[:3]
def multi_query_retrieve(query: str):
    queries = generate_queries(query)
    all_docs = []
    for q in queries:
        docs = dense_retriever.invoke(q)
        all_docs.extend(docs)
    unique_docs = {}
    for doc in all_docs:
        key = (
            doc.metadata.get("source", ""),
            doc.page_content
        )
        unique_docs[key] = doc
    return list(unique_docs.values())
