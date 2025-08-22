import google.generativeai as genai

# Configure API Key
genai.configure(api_key="AIzaSyD-qTyix3U7C267Pd4RnJvmvwWEXMmqXic")

# Initialize Gemini model
model = genai.GenerativeModel("gemini-pro")

# Ask a simple medical question
response = model.generate_content("What is the use of paracetamol?")
print(response.text)
