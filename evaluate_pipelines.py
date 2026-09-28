import os
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
load_dotenv()
judge_llm = ChatOpenAI(
    model="nvidia/nemotron-3.5-lightning:free",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)
def evaluate_answer(query, answer, contexts):
    context_text = "\n\n".join(contexts)
    prompt = f"""
You are evaluating a RAG system.
User Query:
{query}
Retrieved Context:
{context_text}
Generated Answer:
{answer}
Evaluate the answer.Give scores from 0 to 1 for:
    1. relevance
    2. faithfulness
    3. completeness
Return ONLY JSON:
{{
    "relevance": 0.0,
    "faithfulness": 0.0,
    "completeness": 0.0
}}
"""
    response = judge_llm.invoke(prompt)
    text = response.content.strip()
    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        result = {
            "relevance": 0.0,
            "faithfulness": 0.0,
            "completeness": 0.0
        }
    return result
def calculate_score(metrics):
    return (
        0.4 * metrics["relevance"]
        + 0.4 * metrics["faithfulness"]
        + 0.2 * metrics["completeness"]
    )
