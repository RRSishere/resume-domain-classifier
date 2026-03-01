import streamlit as st
import torch
import pickle
import re
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from pypdf import PdfReader
import docx
import pandas as pd

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Resume Domain Classifier",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ================= CUSTOM CSS (PRO UI) =================
st.markdown("""
<style>
.main {
    background-color: #0f172a;
}
h1, h2, h3 {
    color: white;
}
.card {
    background-color:#1e293b;
    padding:20px;
    border-radius:12px;
    box-shadow:0px 4px 15px rgba(0,0,0,0.3);
    margin-bottom:15px;
}
.skill-badge {
    display:inline-block;
    padding:6px 12px;
    margin:4px;
    border-radius:20px;
    background:#4f46e5;
    color:white;
    font-size:13px;
}
</style>
""", unsafe_allow_html=True)

# ================= CONFIG =================
MODEL_PATH = "./BERT_resume_model"

# ================= SKILLS =================
SKILL_KEYWORDS = [
"python","java","c++","javascript","react","node","express",
"mongodb","mysql","html","css","tensorflow","pytorch",
"machine learning","deep learning","nlp","aws","docker",
"kubernetes","git","linux","fastapi","django","flask",
"power bi","tableau","excel","pandas","numpy"
]

def extract_skills(text):
    text_low = text.lower()
    found = [s.upper() for s in SKILL_KEYWORDS if s in text_low]
    return sorted(set(found))

# ================= LOAD MODEL =================
@st.cache_resource
def load_all():
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH,
        low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    with open(f"{MODEL_PATH}/label_encoder.pkl","rb") as f:
        le = pickle.load(f)

    model.eval()
    return model, tokenizer, le

model, tokenizer, le = load_all()

# ================= CLEANING =================
def clean_resume(text):
    text = str(text)
    text = re.sub(r"http\S+|\S+@\S+", " ", text)
    text = re.sub(r"\+?\d[\d\s\-]{8,}", " ", text)
    text = text.replace("\n"," ")
    text = re.sub(r"\s+"," ",text)

    first_line = text[:150]
    text = "TITLE: " + first_line + " BODY: " + text

    return text[:6000]

# ================= TEXT EXTRACTION =================
def extract_text(file):
    name = file.name.lower()

    if name.endswith(".pdf"):
        reader = PdfReader(file)
        text = " ".join([p.extract_text() or "" for p in reader.pages])
        return text,"PDF"

    if name.endswith(".docx"):
        d = docx.Document(file)
        text = " ".join([p.text for p in d.paragraphs])
        return text,"DOCX"

    if name.endswith(".txt"):
        return file.read().decode("utf-8"),"TXT"

    return "","UNKNOWN"

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
    topk = torch.topk(probs,3)

    labels = le.inverse_transform(topk.indices.tolist())
    scores = topk.values.tolist()

    return list(zip(labels,scores))

# ================= HERO HEADER =================
st.markdown("""
<h1 style='text-align:center;'>AI Resume Domain Classification System</h1>
<p style='text-align:center;color:#94a3b8;font-size:18px;'>
Upload Resume → NLP Processing → BERT Prediction → Skill Extraction
</p>
""", unsafe_allow_html=True)

st.divider()

# ================= LAYOUT =================
left, right = st.columns([2,1])

# ---------- LEFT ----------
with left:

    st.markdown("<div class='card'>", unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload Resume (PDF / DOCX / TXT)",
        type=["pdf","docx","txt"]
    )

    if uploaded:

        raw_text, ftype = extract_text(uploaded)
        skills = extract_skills(raw_text)

        st.success(f"Detected Format: {ftype}")

        st.subheader("Resume Preview")
        st.text_area("", raw_text[:1500], height=260)

        results = predict_top3(raw_text)
        best_label, best_score = results[0]

    st.markdown("</div>", unsafe_allow_html=True)

# ---------- RIGHT ----------
with right:

    if uploaded:

        # Prediction Card
        st.markdown("<div class='card'>", unsafe_allow_html=True)

        st.subheader("Prediction")

        st.metric(
            label="Predicted Domain",
            value=best_label,
            delta=f"{best_score*100:.1f}% confidence"
        )

        st.progress(best_score)

        st.markdown("</div>", unsafe_allow_html=True)

        # Chart Card
        st.markdown("<div class='card'>", unsafe_allow_html=True)

        st.subheader("Top Domain Scores")

        df = pd.DataFrame(results, columns=["Domain","Score"])
        st.bar_chart(df.set_index("Domain"))

        st.markdown("</div>", unsafe_allow_html=True)

        # Skills Card
        st.markdown("<div class='card'>", unsafe_allow_html=True)

        st.subheader("Detected Skills")

        if skills:
            badges = "".join(
                [f"<span class='skill-badge'>{s}</span>" for s in skills]
            )
            st.markdown(badges, unsafe_allow_html=True)
        else:
            st.write("No known skills detected")

        st.markdown("</div>", unsafe_allow_html=True)
