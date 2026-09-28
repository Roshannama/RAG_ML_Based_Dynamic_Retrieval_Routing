import re
import numpy as np
ENGINEERED_FEATURES = [
    "word_count",
    "char_count",
    "sentence_count",
    "avg_word_length",
    "number_count",
    "contains_number",
    "question_mark",
    "starts_what",
    "starts_how",
    "starts_why",
    "starts_when",
    "starts_where",
    "exact_keyword_count",
    "compare_keyword_count",
    "process_keyword_count",
    "policy_keyword_count",
    "explain_keyword_count",
    "uppercase_token_count",
    "special_char_count",
]
def extract_query_features(query: str):
    query = str(query)
    lower_query = query.lower().strip()
    words = re.findall(r"\b\w+\b", lower_query)
    sentences = [
        sentence
        for sentence in re.split(r"[.!?]+", query)
        if sentence.strip()
    ]
    numbers = re.findall(
        r"\d+(?:\.\d+)?",
        query
    )
    avg_word_length = (
        np.mean([len(word) for word in words])
        if words
        else 0
    )
    exact_keywords = [
        "exact",
        "specific",
        "precisely",
        "code",
        "id",
        "identifier",
        "number"
    ]
    compare_keywords = [
        "compare",
        "comparison",
        "versus",
        "vs",
        "difference",
        "different"
    ]
    process_keywords = [
        "how",
        "steps",
        "procedure",
        "process",
        "method",
        "way"
    ]
    policy_keywords = [
        "policy",
        "rule",
        "terms",
        "eligibility",
        "requirement"
    ]
    explain_keywords = [
        "explain",
        "why",
        "reason",
        "describe"
    ]
    def keyword_count(keywords):
        return sum(
            lower_query.count(keyword)
            for keyword in keywords
        )
    uppercase_token_count = sum(
        1
        for token in query.split()
        if len(token) > 1 and token.isupper()
    )
    special_char_count = len(
        re.findall(
            r"[^A-Za-z0-9\s]",
            query
        )
    )
    features = {
        "word_count":
            len(words),
        "char_count":
            len(query),
        "sentence_count":
            max(1, len(sentences)),
        "avg_word_length":
            float(avg_word_length),
        "number_count":
            len(numbers),
        "contains_number":
            int(bool(numbers)),
        "question_mark":
            int("?" in query),
        "starts_what":
            int(lower_query.startswith("what")),
        "starts_how":
            int(lower_query.startswith("how")),
        "starts_why":
            int(lower_query.startswith("why")),
        "starts_when":
            int(lower_query.startswith("when")),
        "starts_where":
            int(lower_query.startswith("where")),
        "exact_keyword_count":
            keyword_count(exact_keywords),
        "compare_keyword_count":
            keyword_count(compare_keywords),
        "process_keyword_count":
            keyword_count(process_keywords),
        "policy_keyword_count":
            keyword_count(policy_keywords),
        "explain_keyword_count":
            keyword_count(explain_keywords),
        "uppercase_token_count":
            uppercase_token_count,
        "special_char_count":
            special_char_count,
    }
    return features
