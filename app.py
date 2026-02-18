import streamlit as st
import torch
import pickle
import re
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from pypdf import PdfReader
import docx

# ================= CONFIG =================

MODEL_PATH = "./BERT_resume_model"

# ================= SKILL EXTRACT =================

SKILL_KEYWORDS = [

# ---------- Programming ----------
"python","java","c","c++","c#","go","rust","scala","kotlin","swift",
"javascript","typescript","php","ruby","matlab","r",

# ---------- Frontend ----------
"html","css","sass","less","bootstrap","tailwind",
"react","reactjs","nextjs","vue","vuejs","angular",
"redux","jquery",

# ---------- Backend ----------
"node","nodejs","express","nestjs","django","flask","fastapi",
"spring","spring boot","laravel",".net","asp.net",

# ---------- Mobile ----------
"android","ios","react native","flutter","xamarin",

# ---------- Databases ----------
"mysql","postgresql","postgres","mongodb","redis","sqlite",
"oracle","sql server","cassandra","dynamodb","firebase",

# ---------- Data / ML / AI ----------
"machine learning","deep learning","nlp","computer vision",
"pytorch","tensorflow","keras","sklearn","scikit-learn",
"pandas","numpy","scipy","xgboost","lightgbm",
"data analysis","data science","feature engineering",
"bert","transformer","llm","huggingface",

# ---------- Big Data ----------
"hadoop","spark","pyspark","hive","kafka","airflow",

# ---------- Cloud ----------
"aws","azure","gcp","google cloud",
"ec2","s3","lambda","cloudwatch",
"azure functions","bigquery",

# ---------- DevOps ----------
"docker","kubernetes","helm",
"ci/cd","jenkins","github actions","gitlab ci",
"terraform","ansible",

# ---------- Tools ----------
"git","github","gitlab","bitbucket",
"linux","unix","bash","shell scripting",
"postman","swagger",

# ---------- APIs ----------
"rest api","graphql","fastapi",

# ---------- Testing ----------
"unit testing","pytest","jest","selenium","cypress",

# ---------- Analytics ----------
"power bi","tableau","excel","looker",

# ---------- Security ----------
"oauth","jwt","cyber security","penetration testing"
]

def extract_skills(text):
    text_low = text.lower()
    found = []
    for skill in SKILL_KEYWORDS:
        if skill in text_low:
            found.append(skill.upper())
    return sorted(set(found))

# ================= LOAD MODEL (LOW MEMORY) =================

def load_all():
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH,
        low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    with open(f"{MODEL_PATH}/label_encoder.pkl", "rb") as f:
        le = pickle.load(f)

    model.eval()
    return model, tokenizer, le

model, tokenizer, le = load_all()

# ================= SAME CLEANING AS TRAINING =================

def clean_resume(text):
    text = str(text)

    text = re.sub(r"http\S+|\S+@\S+", " ", text)
    text = re.sub(r"\+?\d[\d\s\-]{8,}", " ", text)

    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)

    first_line = text[:150]
    text = "TITLE: " + first_line + " BODY: " + text

    return text[:6000]

# ================= FILE TEXT EXTRACTION =================

def extract_text(file):
    name = file.name.lower()

    if name.endswith(".pdf"):
        reader = PdfReader(file)
        text = " ".join([p.extract_text() or "" for p in reader.pages])
        return text, "PDF"

    if name.endswith(".docx"):
        d = docx.Document(file)
        text = " ".join([p.text for p in d.paragraphs])
        return text, "DOCX"

    if name.endswith(".txt"):
        return file.read().decode("utf-8"), "TXT"

    return "", "UNKNOWN"

# ================= PREDICTION =================

def predict_top3(text):

    text = clean_resume(text)

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)[0]
    topk = torch.topk(probs, 3)

    labels = le.inverse_transform(topk.indices.tolist())
    scores = topk.values.tolist()

    return list(zip(labels, scores))

# ================= UI =================

st.set_page_config(
    page_title="Resume Domain Classification System",
    layout="wide"
)

st.title("📄 Automated Resume Domain Classification System")
st.caption("Upload resume → Preprocess → BERT Model → Domain Prediction")

left, right = st.columns([2,1])

# ---------- LEFT PANEL ----------

with left:

    uploaded = st.file_uploader(
        "Upload Resume File",
        type=["pdf", "docx", "txt"]
    )

    if uploaded:

        raw_text, ftype = extract_text(uploaded)

        st.success(f"Detected Format: {ftype}")

        skills = extract_skills(raw_text)

        st.subheader("Extracted Text Preview")
        st.text_area(
            "",
            raw_text[:1500],
            height=220
        )

        results = predict_top3(raw_text)
        best_label, best_score = results[0]

# ---------- RIGHT PANEL ----------

with right:

    if uploaded:

        st.subheader("📊 Prediction Results")

        st.metric(
            "Predicted Domain",
            best_label,
            f"{best_score*100:.1f}% confidence"
        )

        st.progress(int(best_score * 100))

        st.subheader("Top 3 Domain Scores")

        for i, (lab, sc) in enumerate(results, 1):
            st.write(f"{i}. {lab} — {sc*100:.2f}%")

        if best_score < 0.50:
            st.warning("Low confidence — manual review recommended")

        # -------- SKILLS UI --------

        st.subheader("🧠 Detected Skills")

        if skills:
            st.write(", ".join(skills))
        else:
            st.write("No known skills detected")
