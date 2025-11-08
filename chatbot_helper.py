# chatbot_helper.py
import os
import streamlit as st

def generate_advice(plant, disease):
    # Check if API key is available
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return f"⚠️ OpenAI API key not configured. Please add OPENAI_API_KEY to your Streamlit secrets. For now, here's basic advice for {disease} in {plant}: Remove affected leaves, improve air circulation, and avoid overwatering."
    
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        prompt = f"""
        You are a friendly gardening assistant.
        The user has a {plant} plant.
        Detected issue: {disease}.
        In 2-3 sentences, explain what this disease is.
        Then give 3-4 simple home treatment tips in bullet points.
        Keep the response concise and practical.
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200
        )

        return response.choices[0].message.content.strip()
    
    except ImportError:
        return "❌ OpenAI package not available. Please check your requirements.txt"
    except Exception as e:
        return f"❌ Error getting AI advice: {str(e)}. Basic advice: Remove affected leaves and improve plant care conditions."
