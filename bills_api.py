from flask import Flask, jsonify, request, abort
from flask_cors import CORS
import json

app = Flask(__name__)
CORS(app)

# Load the bills data
with open("bills.json", "r") as f:
    bills = json.load(f)

# Add IDs to bills if not present
for idx, bill in enumerate(bills):
    bill["id"] = str(idx + 1)


@app.route("/api/bills", methods=["GET"])
def get_bills():
    # Get query parameters
    house = request.args.get("house", "").lower()
    year = request.args.get("year", "")
    title = request.args.get("title", "").lower()

    filtered_bills = bills

    if house:
        filtered_bills = [b for b in filtered_bills if b["house"].lower() == house]
    if year:
        filtered_bills = [b for b in filtered_bills if year in b["date"]]
    if title:
        filtered_bills = [b for b in filtered_bills if title in b["title"].lower()]

    return jsonify(filtered_bills)


@app.route("/api/bills/<bill_id>", methods=["GET"])
def get_bill(bill_id):
    bill = next((b for b in bills if b["id"] == bill_id), None)
    if not bill:
        abort(404)
    return jsonify(bill)


@app.route("/api/bills/search", methods=["POST"])
def search_bills():
    query = request.json.get("query", "").lower()
    results = []

    for bill in bills:
        if query in bill["title"].lower() or query in bill["description"].lower():
            results.append(bill)

    return jsonify(results)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=3000, debug=True)
