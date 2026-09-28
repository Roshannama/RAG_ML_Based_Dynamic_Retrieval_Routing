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
    Write a short hypothetical answer to the following question.
    Do not say that you don't know the answer.
    Generate text that resembles information that might appear
    in a relevant document.
    Generate  relevant information for the following user question,containing more detail about that question hypothetical answer,
    if possible use  full forms or other similar terms to describe.
    Question:
    {query}
    """
)
def generate_hypothetical_document(query: str):
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({
        "query": query
    })
def hyde_retrieve(query: str):
    hypothetical_document = generate_hypothetical_document(query)
    docs = dense_retriever.invoke(
        hypothetical_document
    )
    return docs
