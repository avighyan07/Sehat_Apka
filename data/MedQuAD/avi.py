import streamlit as st
import os
import xml.etree.ElementTree as ET
import pandas as pd
import nltk
from nltk import word_tokenize, pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

# 🔧 Initialize OpenAI client
client = OpenAI()
# If not set in your environment, uncomment below and set your key
# import os
# os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY_HERE"

# Download NLTK data (only first time)
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')

# Function to parse all MedQuAD XML files in all folders
@st.cache_data
def load_medquad_all(base_path):
    data = []
    folders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
    for folder in folders:
        folder_path = os.path.join(base_path, folder)
        files = [f for f in os.listdir(folder_path) if f.endswith('.xml')]
        for file in files:
            file_path = os.path.join(folder_path, file)
            tree = ET.parse(file_path)
            root = tree.getroot()
            for qa in root.findall('.//QAPair'):
                q_elem = qa.find('Question')
                a_elem = qa.find('Answer')
                q = q_elem.text.strip() if q_elem is not None and q_elem.text is not None else ""
                a = a_elem.text.strip() if a_elem is not None and a_elem.text is not None else ""
                if q and a:
                    data.append({"question": q, "answer": a})
    df = pd.DataFrame(data)
    return df

# Named Entity Recognition (basic: nouns as entities)
def named_entity_recognition(text):
    tokens = word_tokenize(text)
    tags = pos_tag(tokens)
    entities = [word for word, tag in tags if tag.startswith('NN')]
    return entities

# Retrieve most relevant answer using TF-IDF cosine similarity
def retrieve_answer(user_q, questions, answers):
    vectorizer = TfidfVectorizer().fit([user_q] + questions)
    vectors = vectorizer.transform([user_q] + questions)
    cosine_sim = cosine_similarity(vectors[0:1], vectors[1:]).flatten()
    max_idx = cosine_sim.argmax()
    max_sim = cosine_sim[max_idx]
    if max_sim > 0.2:
        return answers[max_idx], max_sim
    else:
        return "No relevant answer found.", max_sim

# Medical Report Simplifier function using LLM
def explain_medical_term(term):
    prompt = f"Explain '{term}' in simple terms for a patient. Avoid using complex medical words. Keep it short and clear."
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# Streamlit app UI
st.title("🩺 Medical Q&A Chatbot (MedQuAD) + Report Simplifier")

# Load dataset
st.write("Loading dataset...")
df = load_medquad_all(r"C:\Users\Arunava Chakraborty\Desktop\ChatBots\Medical Q&A Chatbot\data\MedQuAD")
st.write(f"✅ Dataset loaded with {len(df)} question-answer pairs.")
st.write("Sample question preview:", df.iloc[0]['question'])

# User input
user_input = st.text_input("💬 Ask your medical question here:")

if user_input:
    # Entity recognition
    entities = named_entity_recognition(user_input)

    # Retrieve and display answer
    answer, similarity = retrieve_answer(user_input, df['question'].tolist(), df['answer'].tolist())

    st.write("### 📝 Answer:")
    if answer.strip() and "No relevant answer found." not in answer:
        st.success(answer)
    else:
        st.warning("⚠️ Sorry, no suitable answer found for your query.")

    st.write(f"🔗 **Similarity Score:** {similarity:.2f}")

    # Medical Report Simplifier: Explain identified terms
    if entities:
        st.write("### 💡 Layman Explanations for Identified Terms:")
        for ent in entities:
            explanation = explain_medical_term(ent)
            st.write(f"🔹 **{ent}**: {explanation}")
