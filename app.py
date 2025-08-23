from flask import Flask, render_template, request, session
from flask_session import Session
import pandas as pd
import pickle
import google.generativeai as genai
from dotenv import load_dotenv
import os

# Initialize app
app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)



load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


# Load model and vectorizer
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Load dataset (if needed for future enhancements)
df = pd.read_csv(r"C:\Users\dhanu\Desktop\test\Copy of drugsComTrain_raw - drugsComTrain_raw.csv")
df.dropna(subset=["drugName"], inplace=True)

@app.route("/", methods=["GET", "POST"])
def home():
    recommendation = None
    chatbot_answer = None

    if 'chat_history' not in session:
        session['chat_history'] = []

    if request.method == "POST":
        symptom = request.form.get("symptom")
        age = request.form.get("age")
        gender = request.form.get("gender")
        severity = request.form.get("severity")
        question = request.form.get("question")

        # Predict drug
        if symptom:
            vec = vectorizer.transform([symptom])
            prediction = model.predict(vec)[0]
            recommendation = prediction

        # Handle chat
        if question and question.strip():
            try:
                model_gemini = genai.GenerativeModel("models/gemini-1.5-flash")
                response = model_gemini.generate_content(question)
                chatbot_answer = response.text

                session['chat_history'].append((question, chatbot_answer))
                session['chat_history'] = session['chat_history'][-5:]
                session.modified = True
            except Exception as e:
                chatbot_answer = f"⚠️ Gemini Error: {str(e)}"

    return render_template("index.html", recommendation=recommendation,
                           chatbot_answer=chatbot_answer,
                           chat_history=session.get("chat_history", []))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

