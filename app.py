from flask import Flask, jsonify, request
from flask_cors import CORS
import os
from dotenv import load_dotenv
import libsql_experimental as libsql
from generate_embedding import generate_embedding, get_openai_client

# Load environment variables
load_dotenv()
TURSO_DATABASE_URL = os.getenv("TURSO_DATABASE_URL")
TURSO_API_TOKEN = os.getenv("TURSO_API_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Debugging: Check if environment variables are loaded correctly
print("TURSO_DATABASE_URL:", TURSO_DATABASE_URL)
print("TURSO_API_TOKEN:", TURSO_API_TOKEN)
print("OPENAI_API_KEY:", OPENAI_API_KEY)

# Initialize Flask app and OpenAI client
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Adjust origins as needed

conn = libsql.connect(TURSO_DATABASE_URL, auth_token=TURSO_API_TOKEN)
openai_client = get_openai_client()

@app.route('/', methods=['GET'])
def home():
    return "Hello from Flask!", 200

@app.route('/search', methods=['POST'])
def search_properties():
    try:
        # Create a new connection for this request
        conn = libsql.connect(TURSO_DATABASE_URL, auth_token=TURSO_API_TOKEN)

        # Parse incoming JSON request
        data = request.get_json()
        beds = data.get("beds")
        max_price = data.get("max_price")
        description = data.get("description")

        if not beds or not max_price or not description:
            return jsonify({"error": "Missing required parameters"}), 400

        # Step 1: Filter properties by beds and max price
        query = """
            SELECT id, property_name, property_link, image_link
            FROM listings
            WHERE bedrooms = ? AND max_price <= ?;
        """
        result = conn.execute(query, (beds, max_price)).fetchall()

        # Step 2: Convert the result to a list of dictionaries
        properties = [
            {
                "id": row[0],
                "property_name": row[1],
                "property_link": row[2],
                "image_link": row[3]
            }
            for row in result
        ]

        # Step 3: Generate an embedding for the user's description
        user_embedding = generate_embedding(openai_client, description)

        # Step 4: Perform vector similarity search using Turso's `vector_distance_cos`
        similarity_query = f"""
            SELECT id, property_name, property_link, image_link, vector_distance_cos(embedding, vector32('{user_embedding}')) AS similarity
            FROM listings
            WHERE bedrooms = ? AND max_price <= ?
            ORDER BY similarity ASC
            LIMIT 5;
        """
        similar_properties = conn.execute(similarity_query, (beds, max_price)).fetchall()

        # Step 5: Format the result
        top_properties = [
            {
                "id": row[0],
                "property_name": row[1],
                "property_link": row[2],
                "image_link": row[3]
            }
            for row in similar_properties
        ]

        # Return the top 5 similar properties
        return jsonify({"result": top_properties}), 200

    except Exception as e:
        print("Error executing search:", e)
        return jsonify({"error": "Failed to execute search"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
