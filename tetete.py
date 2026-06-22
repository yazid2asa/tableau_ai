from google import genai

client = genai.Client(api_key="")

response = client.models.generate_content(
    model="gemini-3.1-flash-lite",
    contents="Explain how AI works in a few words"
)
print(response.text)