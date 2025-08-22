import pandas as pd

# Load your dataset
df = pd.read_csv("Copy of drugsComTrain_raw - drugsComTrain_raw.csv")

# Drop missing conditions
df = df.dropna(subset=["condition"])

# Get unique sorted list of conditions
conditions = sorted(df["condition"].unique())

# Generate <option> HTML tags
with open("symptom_options.txt", "w", encoding="utf-8") as f:
    f.write('<option value="">-- Select a Symptom --</option>\n')
    for condition in conditions:
        f.write(f'<option value="{condition}">{condition}</option>\n')

print("✅ Symptom dropdown list saved to 'symptom_options.txt'")
