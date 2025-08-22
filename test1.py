import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load dataset
df = pd.read_csv(r"C:\Users\dhanu\Desktop\test\Copy of drugsComTrain_raw - drugsComTrain_raw.csv")  # Rename your file if needed
df = df.dropna(subset=['condition'])

# Train vectorizer on symptoms (you can also use 'reviews' or combine fields for better generalization)
X = df['condition'].astype(str)
y = df['condition'].astype(str)  # predicting the condition back from user symptom

vectorizer_disease = CountVectorizer()
X_vec = vectorizer_disease.fit_transform(X)

model_disease = MultinomialNB()
model_disease.fit(X_vec, y)

# Save the model and vectorizer
with open("disease_model.pkl", "wb") as f:
    pickle.dump(model_disease, f)

with open("disease_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer_disease, f)

print("Disease model trained and saved ✅")
