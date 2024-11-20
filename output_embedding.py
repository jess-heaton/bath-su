# output_embedding.py

import json
import os
from generate_embedding import get_openai_client, generate_embedding

def save_embedding_to_file(embedding, filename):
    """
    Saves the embedding to a file in JSON format.

    Parameters:
        embedding (list): The embedding to save.
        filename (str): The filename to save the embedding to.
    """
    try:
        with open(filename, 'w') as f:
            json.dump(embedding, f, indent=4)
        print(f"Embedding successfully saved to {filename}")
    except Exception as e:
        print(f"Error saving embedding to file: {e}")
        raise e

def main():
    """
    Main function to input description and save its embedding to a file.
    """
    description = input("Enter the description: ").strip()
    if not description:
        print("Description cannot be empty.")
        return

    try:
        client = get_openai_client()
    except ValueError as ve:
        print(ve)
        return
    except Exception as e:
        print(f"Unexpected error initializing OpenAI client: {e}")
        return

    print("Generating embedding...")
    try:
        embedding = generate_embedding(client, description)
    except Exception as e:
        print(f"Failed to generate embedding: {e}")
        return

    # Define the output filename. You can modify this as needed.
    output_filename = "embedding.json"

    try:
        save_embedding_to_file(embedding, output_filename)
    except Exception as e:
        print(f"Failed to save embedding: {e}")

if __name__ == "__main__":
    main()
