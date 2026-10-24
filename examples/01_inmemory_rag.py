"""
Pattern 01: In-Memory Knowledge Base RAG
Demonstrates the fundamental mechanics of Retrieval-Augmented Generation (RAG):
1. In-memory structured knowledge base.
2. Keyword-scoring lexical retrieval algorithm.
3. Strict zero-hallucination prompt construction.
4. Grounded answer generation using Cohere ClientV2.
"""

import os
import sys
import cohere
from dotenv import load_dotenv

# Ensure UTF-8 output on Windows consoles
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

load_dotenv()
api_key = os.getenv("COHERE_API_KEY")
if not api_key:
    raise ValueError("COHERE_API_KEY not found. Please set it in your .env file.")

co = cohere.ClientV2(api_key=api_key)

# 1. In-memory Knowledge Base
knowledge_base = [
    {"id": 1, "text": "The tallest penguins are Emperor penguins."},
    {"id": 2, "text": "Emperor penguins live in Antarctica."},
    {"id": 3, "text": "Penguins are flightless birds found in the Southern Hemisphere."},
    {"id": 4, "text": "The average height of Emperor penguins is about 45 inches."},
    {"id": 5, "text": "Emperor penguins can swim underwater for up to 22 minutes."},
    {"id": 6, "text": "A group of penguins in the water is called a raft."},
    {"id": 7, "text": "Penguins face a long, slow walk inland across the ice to the colony."},
    {"id": 8, "text": "Emperor penguins can live up to 50 years in captivity."},
    {"id": 9, "text": "Emperor penguins are the deepest diving birds; one was recorded diving 565 m deep."},
    {"id": 10, "text": "Sea horses swim vertically with rapid fin movements but are poor swimmers."}
]

# 2. Lexical Keyword Retrieval Function
def retrieve_documents(question: str, k: int = 5) -> list[str]:
    keywords = question.lower().split()
    scored_docs = []

    for doc in knowledge_base:
        score = sum(
            1 for word in keywords
            if word.strip("?,.!") in doc["text"].lower()
        )
        if score > 0:
            scored_docs.append((score, doc["text"]))

    scored_docs.sort(key=lambda item: item[0], reverse=True)
    return [doc for score, doc in scored_docs[:k]]

# 3. Grounded Answer Generation
def generate_answer(question: str, documents: list[str]) -> str:
    context = "\n".join(documents)
    prompt = f"""
Answer ONLY from the context below.

If the answer is not present in the context, reply:
'I couldn't find this information in the supplied context.'

Context:
{context}

Question:
{question}

Answer:
"""
    response = co.chat(
        model="command-a-03-2025",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2
    )

    answer = ""
    for item in response.message.content:
        if hasattr(item, "text"):
            answer += item.text

    return answer.strip()


if __name__ == "__main__":
    question = (
        "How long do Emperor penguins live in captivity and "
        "how deep do they dive? Can they be friends with humans?"
    )

    print("=" * 70)
    print(f"Question: {question}")
    print("=" * 70)

    retrieved = retrieve_documents(question, k=4)
    print(f"\n[Retrieved {len(retrieved)} Context Documents]:")
    for idx, doc in enumerate(retrieved, 1):
        print(f"  {idx}. {doc}")

    answer = generate_answer(question, retrieved)
    print("\n[Grounded LLM Answer]:")
    print(answer)
