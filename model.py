import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
import re

# Preprocess function
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

# Expanded example data
data = [
    {"symptoms": "chest pain", "specialization": "Cardiologist"},
    {"symptoms": "shortness of breath", "specialization": "Cardiologist"},
    {"symptoms": "rash", "specialization": "Dermatologist"},
    {"symptoms": "skin irritation", "specialization": "Dermatologist"},
    {"symptoms": "fever", "specialization": "General Physician"},
    {"symptoms": "high temperature", "specialization": "General Physician"},
    {"symptoms": "headache", "specialization": "Neurologist"},
    {"symptoms": "migraine", "specialization": "Neurologist"},
    {"symptoms": "stomach ache", "specialization": "Gastroenterologist"},
    {"symptoms": "belly pain", "specialization": "Gastroenterologist"},
    {"symptoms": "nausea", "specialization": "Gastroenterologist"},
    {"symptoms": "joint pain", "specialization": "Orthopedist"},
    {"symptoms": "bone fracture", "specialization": "Orthopedist"}
]

# Prepare data
symptoms = [preprocess(item['symptoms']) for item in data]
specializations = [item['specialization'] for item in data]

# Encode specializations using LabelEncoder
le = LabelEncoder()
y = le.fit_transform(specializations)

# Train model
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(symptoms)
model = MultinomialNB()
model.fit(X, y)

# Save model, vectorizer, and label encoder
joblib.dump(model, 'model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
joblib.dump(le, 'label_encoder.pkl')  # Saving the LabelEncoder
print("Model and encoders saved!")
