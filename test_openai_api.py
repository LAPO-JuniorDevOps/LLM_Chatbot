import openai
import os

# Fetch the API key from the environment variable
api_key = os.getenv("OPENAI_API_KEY")

# Check if the API key is fetched correctly
if api_key is None:
    print("Error: OpenAI API key not found.")
else:
    print("API Key successfully retrieved.")

# Set the API key for OpenAI
openai.api_key = api_key

# Example API request to test if the key works
try:
    response = openai.completions.create(
        model="o4-mini",  # Use the correct model name, e.g., gpt-4 or gpt-3.5-turbo
        prompt="Hello, OpenAI!",
        max_tokens=50
    )

    print("Response from OpenAI:")
    print(response)

except Exception as e:
    print(f"Error during API request: {e}")
