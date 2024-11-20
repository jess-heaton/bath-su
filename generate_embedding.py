import os
from dotenv import load_dotenv
from openai import OpenAI

def load_api_key():
    """
    Loads the OpenAI API key from the .env file.
    """
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY in your .env file.")
    return api_key

def get_openai_client():
    """
    Initializes and returns the OpenAI client.
    """
    api_key = load_api_key()
    client = OpenAI(api_key=api_key)
    return client

def generate_embedding(client, text):
    """
    Generates a 1536-dimensional embedding for the given text using OpenAI's API.

    Parameters:
        client (OpenAI): The OpenAI client instance.
        text (str): The input text to generate an embedding for.

    Returns:
        list: A list of floats representing the embedding.
    """
    try:
        response = client.embeddings.create(
            input=[text],
            model="text-embedding-ada-002"
        )
        embedding = response.data[0].embedding
        return embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        raise e