"""
Pattern 06: REST API RAG - Recipe Recommendation Engine
Demonstrates:
1. Dynamic live ingestion from DummyJSON Recipes API.
2. Converting complex nested recipe recipes (ingredients, instructions, tags) into documents.
3. Performing contextual recommendation matching based on available ingredients at home.
"""

import os
import sys
import requests
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

API_URL = "https://dummyjson.com/recipes?limit=30"

def fetch_recipe_documents() -> list[str]:
    response = requests.get(API_URL, timeout=15)
    response.raise_for_status()
    data = response.json()
    recipes = data.get("recipes", [])

    documents = []
    for recipe in recipes:
        doc = (
            f"Recipe: {recipe.get('name')}\n"
            f"Calories per Serving: {recipe.get('caloriesPerServing')}\n"
            f"Ingredients: {', '.join(recipe.get('ingredients', []))}\n"
            f"Tags: {', '.join(recipe.get('tags', []))}\n"
        )
        documents.append(doc)
    return documents


if __name__ == "__main__":
    print(f"Fetching recipes from API: {API_URL}")
    recipes = fetch_recipe_documents()
    print(f"Total Recipe Records Retrieved: {len(recipes)}")

    user_query = """
I have:
- Parmesan cheese
- Chicken
- Veggies (bell peppers, cucumber)
- Avocado and mangoes
Which recipes can I make with these ingredients?
"""
    print("\n" + "=" * 70)
    print(f"User Request: {user_query.strip()}")
    print("=" * 70)

    context = "\n---\n".join(recipes)
    prompt = f"""
Context of available recipes:
{context}

User question:
{user_query}

Recommend only recipes from the supplied context that match some or all of the user's ingredients.
State which ingredients match.
"""
    response = co.chat(
        model="command-a-03-2025",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=400
    )

    answer = ""
    for item in response.message.content:
        if hasattr(item, "text"):
            answer += item.text

    print("\n[Cohere Recipe Recommendations]:")
    print(answer.strip())
