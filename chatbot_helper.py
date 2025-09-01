# chatbot_helper.py
import os
from openai import OpenAI
# Load API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_advice(plant, disease):
    prompt = f"""
    You are a friendly gardening assistant.
    The user has a {plant} plant.
    Detected issue: {disease}.
    In 2-3 sentences, explain what this disease is.
    Then give 3-4 simple home treatment tips in bullet points.
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",  # or gpt-4o if available
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200
    )

    return response.choices[0].message.content.strip()
