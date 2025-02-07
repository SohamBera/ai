from flask import Flask, request, jsonify
import firebase_admin
from firebase_admin import credentials, db
from flask_cors import CORS
import joblib
from difflib import get_close_matches
import logging
import re
from datetime import datetime

# Initialize Firebase
cred = credentials.Certificate("D:/Trial project 1/ai doctor simple search/firebase-adminsdk.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://ai-simple-recom-default-rtdb.firebaseio.com'
})

app = Flask(__name__)
CORS(app)

# Load ML model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text


# Symptom to specialization mapping
symptom_map = {
    "chest pain": "Cardiologist",
    "toothache": "Dentist",
    "skin rash": "Dermatologist",
    "hormonal imbalance": "Endocrinologist",
    "stomach ache": "Gastroenterologist",
    "menstrual issues": "Gynecologist",
    "headache": "Neurologist",
    "tumor": "Oncologist",
    "joint pain": "Orthopedist",
    "vision issues": "Ophthalmologist",
    "sore throat": "Otolaryngologist (ENT)",
    "fever in child": "Pediatrician",
    "depression": "Psychiatrist",
    "cough": "Pulmonologist",
    "arthritis": "Rheumatologist",
    "surgery required": "Surgeon",
    "urinary issues": "Urologist",
    "kidney pain": "Nephrologist",
    "anemia": "Hematologist",
    "allergy": "Allergist/Immunologist",
    "imaging required": "Radiologist",
    "lab test diagnosis": "Pathologist",
    "general issues": "General Practitioner (GP)"
}

# User Signup
@app.route('/signup', methods=['POST'])
def signup():
    data = request.json
    name = data.get('name')
    password = data.get('password')
    mobile = data.get('mobile')
    email = data.get('email', None)

    if not name or not password or not mobile:
        return jsonify({'message': 'Name, password, and mobile number are required!'}), 400

    # Save user details in Firebase
    users_ref = db.reference('users')
    if users_ref.child(mobile).get():
        return jsonify({'message': 'User already exists!'}), 409

    users_ref.child(mobile).set({
        'name': name,
        'password': password,
        'email': email
    })

    return jsonify({'message': 'Signup successful!'}), 201

# User Login
@app.route('/login', methods=['POST'])
def login():
    data = request.json
    mobile = data.get('mobile')
    password = data.get('password')

    if not mobile or not password:
        return jsonify({'message': 'Mobile number and password are required!'}), 400

    # Fetch user details from Firebase
    users_ref = db.reference('users')
    user = users_ref.child(mobile).get()

    if user:
        if user['password'] == password:
            return jsonify({'message': 'Login successful!', 'user': user}), 200
        else:
            return jsonify({'message': 'Invalid password!'}), 401
    else:
        return jsonify({'message': 'User not found!'}), 404

# Doctor Recommendation
@app.route('/recommend', methods=['POST'])
def recommend_doctor():
    data = request.json
    symptoms = data.get("symptoms", "").lower()
    logging.debug(f"Received symptoms: {symptoms}")

    if not symptoms:
        return jsonify({'message': 'Symptoms are required!'}), 400

    # ML Model Prediction
    symptom_vector = vectorizer.transform([symptoms])
    predicted_label = model.predict(symptom_vector)[0]
    
    # Decode predicted label to specialization
    predicted_specialization = label_encoder.inverse_transform([predicted_label])[0]
    logging.debug(f"Predicted specialization: {predicted_specialization}")

    # Fallback to manual mapping if needed
    recommended_specialization = predicted_specialization or "General Practitioner (GP)"
    
    # Check manual mapping if ML prediction doesn't match well
    for symptom, specialization in symptom_map.items():
        if symptom in symptoms:
            recommended_specialization = specialization
            break

    # Use fuzzy matching if no direct or ML match found
    closest_match = get_close_matches(symptoms, symptom_map.keys(), n=1, cutoff=0.8)
    if closest_match:
        recommended_specialization = symptom_map[closest_match[0]]

    logging.debug(f"Final recommended specialization: {recommended_specialization}")

    # Fetch doctors from Firebase
    doctors_ref = db.reference('doctors')
    all_doctors = doctors_ref.get()

    if not all_doctors:
        return jsonify({'message': 'No doctors found in the database!'}), 404

    if isinstance(all_doctors, dict):
        all_doctors = list(all_doctors.values())

    # Filter doctors by specialization
    recommended_doctors = [
        doctor for doctor in all_doctors
        if doctor and isinstance(doctor, dict) and doctor.get('specialization') == recommended_specialization
    ]

    if not recommended_doctors:
        return jsonify({'message': f'No {recommended_specialization} available!'}), 404

    return jsonify(recommended_doctors), 200


# Book Appointment
@app.route('/book', methods=['POST'])
def book_appointment():
    data = request.json
    doctor_name = data.get('doctor_name')
    user_mobile = data.get('user_mobile')
    
    if not doctor_name or not user_mobile:
        return jsonify({'message': 'Doctor name, user mobile are required!'}), 400
    
    today_date = datetime.now().strftime('%Y-%m-%d')

    # Reference to bookings in Firebase
    bookings_ref = db.reference('bookings')

    # Query existing bookings for this doctor and time slot
    existing_bookings = bookings_ref.order_by_child('user_mobile').equal_to(user_mobile).get()

    # Optional: Prevent double booking by the same user for the same slot
    for booking in existing_bookings.values():
        if booking['doctor_name'] == doctor_name and booking['date'] == today_date:
            return jsonify({'message': f'You have already booked {doctor_name} today!'}), 400

    # Save new booking
    bookings_ref.push({
        'doctor_name': doctor_name,
        'user_mobile': user_mobile,
        'date': today_date
    })

    return jsonify({
        'message': 'Appointment booked successfully!',

    }), 201

@app.route('/my_appointments', methods=['GET'])
def get_user_appointments():
    user_mobile = request.args.get('user_mobile')
    if not user_mobile:
        return jsonify({'message': 'User mobile is required!'}), 400

    # Get today's date
    today_date = datetime.now().strftime('%Y-%m-%d')

    # Reference to bookings in Firebase
    bookings_ref = db.reference('bookings')

    # Query for the user's appointments on the current day
    appointments = bookings_ref.order_by_child('user_mobile').equal_to(user_mobile).get()

    # Filter appointments for today's date
    today_appointments = {key: value for key, value in appointments.items() if value['date'] == today_date}

    return jsonify(today_appointments), 200
@app.route('/bookings/<doctor_name>', methods=['GET'])
def get_doctor_bookings(doctor_name):
    # Get all bookings for a specific doctor
    bookings_ref = db.reference('bookings')
    doctor_bookings = bookings_ref.order_by_child('doctor_name').equal_to(doctor_name).get()

    # Count bookings per time slot
    time_slot_counts = {}
    for booking in doctor_bookings.values():
        time_slot = booking.get('time_slot')
        time_slot_counts[time_slot] = time_slot_counts.get(time_slot, 0) + 1

    return jsonify(time_slot_counts), 200

@app.route('/bookings_count', methods=['GET'])
def bookings_count():
    doctor_name = request.args.get('doctor_name')
    if not doctor_name:
        return jsonify({"error": "Doctor name is required"}), 400

    # Assuming you're using Firebase Realtime Database
    bookings_ref = db.reference('bookings')
    bookings = bookings_ref.order_by_child('doctor_name').equal_to(doctor_name).get()
    
    count = len(bookings) if bookings else 0
    return jsonify({"count": count})



if __name__ == '__main__':
    app.run(debug=True)
