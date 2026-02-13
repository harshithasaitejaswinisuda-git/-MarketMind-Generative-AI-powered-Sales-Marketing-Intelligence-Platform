from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests, os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# ================= CORS =================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= API KEYS =================
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
GROQ_KEY = os.getenv("GROQ_API_KEY")
HF_KEY = os.getenv("HF_API_KEY")
IBM_KEY = os.getenv("IBM_API_KEY")

class Prompt(BaseModel):
    text: str


# ================= GEMINI =================
def call_gemini(prompt):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={GEMINI_KEY}"
    body = {"contents":[{"parts":[{"text": prompt}]}]}

    res = requests.post(url, json=body)

    if res.status_code != 200:
        raise HTTPException(500, "Gemini API Error")

    data = res.json()

    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except:
        return "No response from Gemini"


# ================= GROQ =================
def call_groq(prompt):
    url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {GROQ_KEY}",
        "Content-Type": "application/json"
    }

    body = {
        "model": "llama3-70b-8192",
        "messages": [{"role": "user", "content": prompt}]
    }

    res = requests.post(url, headers=headers, json=body)

    if res.status_code != 200:
        raise HTTPException(500, "Groq API Error")

    return res.json()["choices"][0]["message"]["content"]


# ================= HUGGING FACE =================
def hf_sentiment(text):
    API_URL = "https://api-inference.huggingface.co/models/distilbert-base-uncased-finetuned-sst-2-english"

    headers = {"Authorization": f"Bearer {HF_KEY}"}

    res = requests.post(API_URL, headers=headers, json={"inputs": text})

    if res.status_code != 200:
        raise HTTPException(500, "HF API Error")

    return res.json()


# ================= IBM WATSON =================
def ibm_analysis(text):
    url = "https://api.us-south.natural-language-understanding.watson.cloud.ibm.com/v1/analyze?version=2022-04-07"

    auth = ("apikey", IBM_KEY)

    data = {
        "text": text,
        "features": {
            "keywords": {"limit": 5},
            "entities": {"limit": 5}
        }
    }

    res = requests.post(url, auth=auth, json=data)

    if res.status_code != 200:
        raise HTTPException(500, "IBM Watson Error")

    return res.json()


# =====================================================
# API ROUTES
# =====================================================

@app.post("/campaign")
def generate_campaign(p: Prompt):
    prompt = f"Create marketing campaign strategy for: {p.text}"
    return {"result": call_gemini(prompt)}


@app.post("/sales-pitch")
def sales_pitch(p: Prompt):
    prompt = f"Write persuasive sales pitch for: {p.text}"
    return {"result": call_groq(prompt)}


@app.post("/lead-score")
def lead_score(p: Prompt):
    return {"score": hf_sentiment(p.text)}


@app.post("/market-analysis")
def market_analysis(p: Prompt):
    return {"analysis": ibm_analysis(p.text)}


@app.post("/business-insights")
def business_insights(p: Prompt):
    prompt = f"Provide business insights for: {p.text}"
    return {"insights": call_gemini(prompt)}
