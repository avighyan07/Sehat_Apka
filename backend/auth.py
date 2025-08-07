from flask import Blueprint, render_template, request, flash, redirect, url_for
import os
from werkzeug.utils import secure_filename
import joblib
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import xml.etree.ElementTree as ET
import pandas as pd
import nltk
from nltk import word_tokenize, pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
import re
# ⚠️ Suppress TensorFlow and warnings for clean console
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)



auth = Blueprint('auth', __name__)

# 📥 Auto-download pneumonia model if not present


import google.generativeai as genai
# --- Load your Gemini API key ---
genai.configure(api_key="AIzaSyAd7wKJiROhKdP1D6AZvL5vTmmkGGUumPE")
gemini_model = genai.GenerativeModel('gemini-1.5-flash')


from keras.models import load_model
model_pneumonia = None  # global cache
model_breast_cancer = None

    global model_pneumonia
    if model_pneumonia is None:
        model_pneumonia = load_model("pneumonia_model.h5")
    return model_pneumonia

def get_breast_cancer_model():
    global model_breast_cancer
    if model_breast_cancer is None:
        model_breast_cancer = load_model('final_CNN.h5')
    return model_breast_cancer
# 🔥 Load ML and DL models globally
model_diabetes = joblib.load('ml_models/model4.joblib')
model_heart = joblib.load('ml_models/model2.joblib')
model_kidney = joblib.load('ml_models/model3.joblib')
# # model_pneumonia = load_model('pneumonia_model.h5')
# model_breast_cancer = load_model('final_CNN.h5')


# 🔥 Load HuggingFace pipeline
# simplifier = pipeline("text2text-generation", model="google/flan-t5-small")





def extract_keywords(text):
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
    stopwords = {
        "what", "does", "mean", "the", "your", "with", "from", "that", "which",
        "have", "how", "and", "when", "then", "than", "into", "this", "about", "like"
    }
    keywords = [word for word in words if word not in stopwords]
    return list(set(keywords))

# Generate answer using Gemini
def generate_answer_with_gemini(question):
    try:
        response = gemini_model.generate_content(question)
        return response.text.strip()
    except Exception as e:
        print(f"Gemini API error: {e}")
        return "Sorry, I'm unable to answer that right now."

# Explain each medical term using Gemini
def explain_medical_term(term):
    try:
        prompt = f"Explain the medical term '{term}' in simple and accurate language for a layperson."
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Error explaining term {term}: {e}")
        return "No explanation available."

# 📁 Routes

@auth.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        image_file = request.files['image']
        if image_file and image_file.filename.endswith(('.png', '.jpg', '.jpeg', '.gif')):
            upload_dir = 'static/uploads/medical_records'
            os.makedirs(upload_dir, exist_ok=True)
            filename = secure_filename(image_file.filename)
            image_path = os.path.join(upload_dir, filename)
            image_file.save(image_path)
            flash('Image uploaded successfully!', category='success')
        else:
            flash('Invalid file type. Please upload an image.', category='error')
    return render_template("upload.html")

@auth.route('/mlpred')
def ml_pred():
    return render_template('mlpred.html')

@auth.route('/dlpred')
def dlpred():
    return render_template('dl_model.html')

@auth.route('/diabetes', methods=['GET', 'POST'])
def diabetes():
    result = None
    if request.method == 'POST':
        input_data = [[
            int(request.form['Pregnancies']),
            int(request.form['Glucose']),
            int(request.form['BloodPressure']),
            int(request.form['SkinThickness']),
            int(request.form['Insulin']),
            float(request.form['BMI']),
            float(request.form['DiabetesPedigreeFunction']),
            int(request.form['Age'])
        ]]
        prediction = model_diabetes.predict(input_data)[0]
        result = '⚠️ Based on the analysis, there are indications of potential diabetes. Please consult a medical professional for a detailed evaluation.' if prediction == 1 else '✅ Based on the analysis, no signs of diabetes were detected. Maintain regular health check-ups to stay updated.'
    return render_template('diabetes_form.html', prediction=result)


@auth.route("/nearby")
def find_doctor():
    return render_template("nearby.html")


@auth.route('/heart', methods=['GET', 'POST'])
def heart_disease():
    result = None
    if request.method == 'POST':
        input_data = [[
            int(request.form['age']), int(request.form['sex']),
            int(request.form['cp']), float(request.form['trestbps']),
            float(request.form['chol']), int(request.form['fbs']),
            int(request.form['restecg']), float(request.form['thalach']),
            int(request.form['exang']), float(request.form['oldpeak']),
            int(request.form['slope']), int(request.form['ca']),
            int(request.form['thal'])
        ]]
        prediction = model_heart.predict(input_data)[0]
        result = '⚠️ Based on the analysis, there are indications of potential heart disease. Please consult a medical professional for a detailed evaluation.' if prediction == 1 else '✅ Based on the analysis, no signs of heart disease were detected. Maintain regular health check-ups to stay updated.'
        print(result)
    return render_template('heart_form.html', prediction=result)

@auth.route('/kidney', methods=['GET', 'POST'])
def kidney_disease():
    result = None
    if request.method == 'POST':
        input_data = [[
            int(request.form['age']), float(request.form['blood_pressure']),
            float(request.form['specific_gravity']), int(request.form['albumin']),
            int(request.form['sugar']), int(request.form['red_blood_cells']),
            int(request.form['pus_cell']), int(request.form['pus_cell_clumps']),
            int(request.form['bacteria']), float(request.form['blood_glucose_random']),
            float(request.form['blood_urea']), float(request.form['serum_creatinine']),
            float(request.form['sodium']), float(request.form['potassium']),
            float(request.form['haemoglobin']), float(request.form['packed_cell_volume']),
            float(request.form['white_blood_cell_count']), float(request.form['red_blood_cell_count']),
            int(request.form['hypertension']), int(request.form['diabetes_mellitus']),
            int(request.form['coronary_artery_disease']), int(request.form['appetite']),
            int(request.form['peda_edema']), int(request.form['aanemia'])
        ]]
        prediction = model_kidney.predict(input_data)[0]
        result = '⚠️ Based on the analysis, there are indications of potential kidney disease. Please consult a medical professional for a detailed evaluation.' if prediction == 1 else '✅ Based on the analysis, no signs of kidney disease were detected. Maintain regular health check-ups to stay updated.'
    return render_template('kidney_form.html', prediction=result)

@auth.route('/pneumonia', methods=['GET', 'POST'])
def pneumonia():
    result = None
    if request.method == 'POST':
        image_file = request.files['xray_image']
        if image_file:
            file_bytes = np.frombuffer(image_file.read(), np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
            if img is None:
                flash('Invalid image.', category='error')
                return redirect(url_for('auth.pneumonia'))

            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            img_resized = cv2.resize(img_rgb, (150, 150)) / 255.0
            img_resized = np.reshape(img_resized, (1, 150, 150, 3))
            prediction = model_pneumonia.predict(img_resized)
            predicted_class = np.argmax(prediction, axis=1)[0]
            result = '⚠️ Based on the analysis, there are indications of potential pneumonia. Please consult a medical professional for a detailed evaluation.' if predicted_class == 1 else '✅ Based on the analysis, no signs of pneumonia were detected. Maintain regular health check-ups to stay updated.'
    return render_template('pneumonia_form.html', result=result)

@auth.route('/breast_cancer', methods=['GET', 'POST'])
def breast_cancer():
    result = None
    if request.method == 'POST':
        image_file = request.files['xray_image']
        if image_file:
            file_bytes = np.frombuffer(image_file.read(), np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
            if img is None:
                flash('Invalid image.', category='error')
                return redirect(url_for('auth.breast_cancer'))

            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

            
            img_resized = cv2.resize(img_rgb, (50, 50)) / 255.0
            img_resized = np.reshape(img_resized, (1, 50, 50, 3))  # Match model input shape

            prediction = model_breast_cancer.predict(img_resized)
            predicted_class = 1 if prediction[0][0] > 0.5 else 0
            result = '⚠️ Based on the analysis, there are indications of potential breast cancer. Please consult a medical professional for a detailed evaluation.' if predicted_class == 1 else '✅ Based on the analysis, no signs of breast cancer were detected. Maintain regular health check-ups to stay updated.'
    return render_template('breast_cancerform.html', result=result)

@auth.route('/medical_qna', methods=['GET', 'POST'])
def medical_qna():
    answer = None
    explanations = {}
    similarity = None  # Optional: in case you plan to add similarity score later

    if request.method == 'POST':
        user_input = request.form['question']
        keywords = extract_keywords(user_input)
        answer = generate_answer_with_gemini(user_input)
        explanations = {word: explain_medical_term(word) for word in keywords}

    return render_template(
        'medical_qna.html',
        answer=answer,
        similarity=similarity,
        explanations=explanations
    )

@auth.route('/nearby')
def nearby():
    return render_template('nearby.html')

@auth.route('/disease_info', methods=['GET', 'POST'])
def disease_info_page():
    diseases = ['diabetes', 'heart', 'kidney', 'pneumonia', 'breast_cancer']
    selected = None

    if request.method == 'POST':
        selected = request.form.get('disease')

    return render_template('disease_info.html', diseases=diseases, selected=selected)

