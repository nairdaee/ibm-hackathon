import os
from flask import Flask, jsonify, request, abort
from flask_cors import CORS
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import openai

app = Flask(__name__)
CORS(app)

# Configure OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")


# Load the bills data
with open("bills.json", "r") as f:
    bills = json.load(f)

# Add IDs to bills if not present
# for idx, bill in enumerate(bills):
#     bill["id"] = str(idx + 1)


def get_embedding(text):
    """Get embedding for text using OpenAI's API"""
    response = openai.embeddings.create(model="text-embedding-ada-002", input=text)
    return response.data[0].embedding


def find_similar_bills(query, bills, top_k=5):
    """Find bills similar to the query using embeddings"""
    query_embedding = get_embedding(query)

    # Convert embeddings to NumPy arrays and print shapes
    query_embedding = np.array(query_embedding, dtype=np.float32)
    bill_embeddings = np.array(
        [bill["embeddings"] for bill in bills if len(bill["embeddings"]) > 0],
        dtype=np.float32,
    )

    # Check if bill_embeddings is 2D
    if bill_embeddings.ndim != 2:
        raise ValueError("bill_embeddings should be a 2D array.")

    # Normalize embeddings
    query_norm = np.linalg.norm(query_embedding)
    bill_norms = np.linalg.norm(bill_embeddings, axis=1)

    # Avoid division by zero
    if query_norm == 0 or np.any(bill_norms == 0):
        raise ValueError(
            "Embeddings contain zero vectors; cannot compute cosine similarity."
        )

    # Compute cosine similarities
    similarities = bill_embeddings @ query_embedding / (bill_norms * query_norm)

    # Pair each bill with its similarity score
    bill_similarity_pairs = list(zip(bills, similarities))

    # Sort the bills by similarity in descending order
    bill_similarity_pairs.sort(key=lambda x: x[1], reverse=True)

    # Return the top_k most similar bills
    return [bill for bill, _ in bill_similarity_pairs[:top_k]]


def compare_bills(bill1, bill2):
    """Compare two bills using OpenAI's GPT-4"""
    # print("Bill 1: ", bill1)
    # print("Bill 2: ", bill2)
    prompt = f"""
    Compare these two bills and give a precise report on them:

    Bill 1: {bill1['title']}
    Description: {bill1['description']}
    Positives: {', '.join([p['title'] + ': ' + p['explanation'] for p in bill1['positives']])}
    Negatives: {', '.join([n['title'] + ': ' + n['explanation'] for n in bill1['negatives']])}

    Bill 2: {bill2['title']}
    Description: {bill2['description']}
    Positives: {', '.join([p['title'] + ': ' + p['explanation'] for p in bill2['positives']])}
    Negatives: {', '.join([n['title'] + ': ' + n['explanation'] for n in bill2['negatives']])}

    Provide a structured comparison with:
    1. Key similarities
    2. Unique aspects of each bill
    3. Common themes or objectives
    """

    # print("Prompt: ", prompt)

    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a legislative analysis expert."},
            {"role": "user", "content": prompt},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "LegislativeComparison",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "bill_1_title": {
                            "type": "string",
                            "description": "Title of the first bill",
                        },
                        "bill_2_title": {
                            "type": "string",
                            "description": "Title of the second bill",
                        },
                        "comparison": {
                            "type": "object",
                            "properties": {
                                "similarities": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "List of key similarities between the bills",
                                },
                                "bill_1_unique": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Unique aspects of the first bill",
                                },
                                "bill_2_unique": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Unique aspects of the second bill",
                                },
                                "common_themes": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Common themes or objectives shared by both bills",
                                },
                            },
                            "required": [
                                "similarities",
                                "bill_1_unique",
                                "bill_2_unique",
                                "common_themes",
                            ],
                            "additionalProperties": False,
                        },
                    },
                    "required": ["bill_1_title", "bill_2_title", "comparison"],
                    "additionalProperties": False,
                },
            },
        },
        temperature=0,
    )

    return json.loads(response.choices[0].message.content)


@app.route("/api/bills", methods=["GET"])
def get_bills():
    # Get query parameters
    house = request.args.get("house", "").lower()
    year = request.args.get("year", "")
    title = request.args.get("title", "").lower()

    filtered_bills = [
        {
            k: v
            for k, v in b.items()
            if k not in ["embeddings", "positives", "negatives"]
        }
        for b in bills
    ]

    if house:
        filtered_bills = [b for b in filtered_bills if b["house"].lower() == house]
    if year:
        filtered_bills = [b for b in filtered_bills if year in b["date"]]
    if title:
        filtered_bills = [b for b in filtered_bills if title in b["title"].lower()]

    return jsonify(filtered_bills)


@app.route("/api/bills/<bill_id>", methods=["GET"])
def get_bill(bill_id):
    # Convert bill_id to integer since it comes as string from URL
    bill_id = int(bill_id)
    bill = next((b for b in bills if b["id"] == bill_id), None)
    if not bill:
        print("Bill wasn't found")
        abort(404)
    bill_without_embeddings = {k: v for k, v in bill.items() if k != "embeddings"}
    return jsonify(bill_without_embeddings)


@app.route("/api/bills/search", methods=["POST"])
def search_bills():
    query = request.json.get("query", "").lower()
    results = []

    for bill in bills:
        if query in bill["title"].lower() or query in bill["description"].lower():
            bill_without_embeddings = {
                k: v for k, v in bill.items() if k != "embeddings"
            }
            results.append(bill_without_embeddings)

    return jsonify(results)


@app.route("/api/bills/compare", methods=["POST"])
def compare_bills_endpoint():
    data = request.json

    # Handle direct bill IDs
    # print(data["bill_id_2"])
    if "bill_id_1" in data and "bill_id_2" in data:
        bill1 = next((b for b in bills if b["id"] == data["bill_id_1"]), None)
        bill2 = next((b for b in bills if b["id"] == data["bill_id_2"]), None)

        if not bill1 or not bill2:
            return jsonify({"error": "One or both bills not found"}), 404

    # Handle natural language queries
    elif "query1" in data and "query2" in data:
        similar_bills1 = find_similar_bills(data["query1"], top_k=1)
        similar_bills2 = find_similar_bills(data["query2"], top_k=1)

        if not similar_bills1 or not similar_bills2:
            return jsonify({"error": "Couldn't find matching bills"}), 404

        bill1 = similar_bills1[0]
        bill2 = similar_bills2[0]

    else:
        return jsonify({"error": "Invalid request format"}), 400

    comparison = compare_bills(bill1, bill2)
    # print(comparison)

    return jsonify(
        {
            "bill_1_title": bill1["title"],
            "bill_2_title": bill2["title"],
            "comparison": comparison,
        }
    )


@app.route("/api/bills/search_similar", methods=["POST"])
def search_similar_bills():
    data = request.json
    query = data.get("query")
    if not query:
        return jsonify({"error": "Query is required"}), 400

    similar_bills = find_similar_bills(query, bills=bills)
    return jsonify(
        [
            {
                "id": bill["id"],
                "title": bill["title"],
                "description": bill["description"],
            }
            for bill in similar_bills
        ]
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000, debug=True)
