import json
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
load_dotenv()
llm = ChatOpenAI(
    model="nvidia/nemotron-3.5-lightning:free",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)
def analyze_query(query):
    prompt = f"""
Analyze the following user query for a RAG system.
Return ONLY valid JSON.
Query:
{query}
Return exactly these fields:
{{
    "query_type": "definition | factual | conceptual | comparison | procedural | multi_hop | troubleshooting",
    "complexity": "low | medium | high",
    "multi_concept": true,
    "keyword_heavy": false,
    "requires_multiple_documents": false,
    "requires_exact_match": false
}}
"""
    response = llm.invoke(prompt)
    text = response.content.strip()
    try:
        features = json.loads(text)
        return {
            "query_type": features.get(
                "query_type",
                "factual"
            ),
            "complexity": features.get(
                "complexity",
                "medium"
            ),
            "multi_concept": bool(
                features.get(
                    "multi_concept",
                    False
                )
            ),
            "keyword_heavy": bool(
                features.get(
                    "keyword_heavy",
                    False
                )
            ),
            "requires_multiple_documents": bool(
                features.get(
                    "requires_multiple_documents",
                    False
                )
            ),
            "requires_exact_match": bool(
                features.get(
                    "requires_exact_match",
                    False
                )
            )
        }
    except json.JSONDecodeError:
        print(
            "\nQuery analyzer returned invalid JSON:"
        )
        print(text)
        return {
            "query_type": "factual",
            "complexity": "medium",
            "multi_concept": False,
            "keyword_heavy": False,
            "requires_multiple_documents": False,
            "requires_exact_match": False
        }
