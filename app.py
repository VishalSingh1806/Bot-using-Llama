from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from datetime import datetime
import sqlite3
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from rapidfuzz import fuzz, process
import logging
import os
import random
import time
from functools import lru_cache

# Predefined openings
OPENINGS = {
    "default": [
        "Here's what I found:",
        "Let me explain:",
        "Absolutely! Here's the answer:",
        "Sure! Here's the information you need:",
    ],
    "fact": [
        "Did you know that...",
        "Here's an interesting fact:",
        "Let me share this with you:",
    ],
    "time": [
        "Oh, this is interesting! It happened in...",
        "Here's the timeline:",
    ],
}

# Fallback Knowledge Base for General Queries
FALLBACK_KB = {
    "what can you do": "I can answer questions about EPR and assist with understanding concepts like plastic waste management, rules, and responsibilities. Try asking something specific!",
    "who made you": "I was developed as a collaborative effort to assist with EPR and related topics using advanced AI capabilities!",
    "how do you work": "I analyze your questions, look up answers in a database, and refine them using an advanced AI model for conversational responses.",
    "can you help me": "Of course! Ask me about EPR, plastic waste management, or any related topics, and I'll do my best to help.",
}

logging.basicConfig(level=logging.DEBUG)

async def lifespan(app: FastAPI):
    logging.info("Application startup: Initializing resources.")
    yield
    logging.info("Application shutdown: Cleaning up resources.")

app = FastAPI(lifespan=lifespan)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_FILES_DIR = os.path.join(CURRENT_DIR, "static")
TEMPLATES_DIR = os.path.join(CURRENT_DIR, "templates")
DB_PATH = os.path.join(CURRENT_DIR, "knowledge_base.db")

app.mount("/static", StaticFiles(directory=STATIC_FILES_DIR), name="static")

hf_token = "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"

@app.get("/")
async def read_root():
    index_file = os.path.join(TEMPLATES_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    raise HTTPException(status_code=404, detail="Frontend index.html not found")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@lru_cache(maxsize=1)
def load_sentence_bert():
    try:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        logging.info("Sentence-BERT model loaded successfully.")
        return model
    except Exception as e:
        logging.error(f"Failed to load Sentence-BERT model: {e}")
        raise RuntimeError(f"Failed to load Sentence-BERT model: {e}")

try:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    llama_tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-2-7b-chat-hf",
        use_auth_token=hf_token
    )
    llama_model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-chat-hf",
        device_map=None,
        torch_dtype=torch.float16 if "cuda" in device else torch.float32,
        low_cpu_mem_usage=True
    ).to(device)

    llama_model.eval()
    logging.info("LLaMA 2 model loaded successfully on %s.", device)
except Exception as e:
    logging.error(f"Failed to load LLaMA 2 model: {e}")
    raise RuntimeError(f"Failed to load LLaMA 2 model: {e}")

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="cache-system uses symlinks by default")

def connect_db():
    try:
        return sqlite3.connect(DB_PATH)
    except sqlite3.Error as e:
        logging.error(f"Database connection error: {e}")
        raise HTTPException(status_code=500, detail="Database connection failed.")

def compute_embedding(text: str):
    model = load_sentence_bert()
    return model.encode(text).reshape(1, -1)

def query_validated_qa(user_embedding):
    try:
        conn = connect_db()
        cursor = conn.cursor()
        cursor.execute("SELECT question, answer, embedding FROM ValidatedQA")
        rows = cursor.fetchall()

        max_similarity = 0.0
        best_answer = None

        for row in rows:
            if len(row) != 3 or not all(row):
                logging.warning(f"Skipping malformed or incomplete row: {row}")
                continue

            question, db_answer, db_embedding = row

            try:
                db_embedding_array = np.frombuffer(db_embedding, dtype=np.float32).reshape(1, -1)
                similarity = cosine_similarity(user_embedding, db_embedding_array)[0][0]
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_answer = db_answer
            except ValueError as e:
                logging.error(f"Error processing embedding for row: {row} - {e}")
                continue

        conn.close()
        if max_similarity >= 0.7:
            return best_answer, max_similarity
        return None, 0.0
    except sqlite3.Error as e:
        logging.error(f"Database query error: {e}")
        return None, 0.0

def fuzzy_match_fallback(question: str) -> str:
    match, score = process.extractOne(question, FALLBACK_KB.keys(), scorer=fuzz.ratio)
    if score >= 80:
        return FALLBACK_KB[match]
    logging.warning(f"No close match found for question: '{question}' (Best match: '{match}' with score {score})")
    return None

def search_sections(query: str):
    try:
        conn = connect_db()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, content FROM Sections WHERE content LIKE ? LIMIT 10;",
            (f"%{query}%",),
        )
        results = cursor.fetchall()
        conn.close()

        return [{"id": row[0], "content": row[1]} for row in results]
    except sqlite3.Error as e:
        logging.error(f"Database error: {e}")
        return []

def get_dynamic_opening(query: str) -> str:
    query = query.lower()
    if any(keyword in query for keyword in ["when", "date", "time", "timeline"]):
        return random.choice(OPENINGS["time"])
    elif any(keyword in query for keyword in ["who", "what", "fact"]):
        return random.choice(OPENINGS["fact"])
    else:
        return random.choice(OPENINGS["default"])

@app.post("/chat")
async def chat_endpoint(request: Request):
    try:
        start_time = time.time()

        data = await request.json()
        question = data.get("message", "").strip().lower()

        if not question:
            raise HTTPException(status_code=400, detail="Message cannot be empty.")

        logging.debug(f"User question: {question}")

        fallback_response = fuzzy_match_fallback(question)
        if fallback_response:
            response_time = time.time() - start_time
            return {
                "answer": fallback_response,
                "confidence": 1.0,
                "source": "fuzzy fallback knowledge base",
                "response_time": f"{response_time:.2f} seconds",
            }

        user_embedding = compute_embedding(question)
        logging.debug(f"Computed user embedding: {user_embedding}")

        answer, confidence = query_validated_qa(user_embedding)
        logging.debug(f"Query result - Answer: {answer}, Confidence: {confidence}")

        if answer and confidence >= 0.8:
            prompt = f"Rephrase this information in a friendly and conversational tone:\n\n{answer}"
            inputs = llama_tokenizer(prompt, return_tensors="pt").to(llama_model.device)
            outputs = llama_model.generate(
                **inputs,
                max_new_tokens=140,
                do_sample=True,
                top_k=50,
                temperature=0.7
            )
            refined_response = llama_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

            opening = get_dynamic_opening(question)
            final_answer = f"{opening} {refined_response}"

            response_time = time.time() - start_time

            return {
                "answer": final_answer,
                "confidence": confidence,
                "source": "database + llama",
                "response_time": f"{response_time:.2f} seconds",
            }

        response_time = time.time() - start_time
        return {
            "answer": "I'm sorry, I couldn't find relevant information. Feel free to ask about EPR or related topics!",
            "confidence": 0.0,
            "source": "fallback response",
            "response_time": f"{response_time:.2f} seconds",
        }

    except Exception as e:
        logging.error(f"Error in /chat endpoint: {e}")
        response_time = time.time() - start_time
        return {
            "answer": "An internal error occurred. Please try again later.",
            "confidence": 0.0,
            "source": "error",
            "response_time": f"{response_time:.2f} seconds",
        }

@app.post("/add")
async def add_to_validated_qa(request: Request):
    try:
        data = await request.json()
        question = data.get("question", "").strip()
        answer = data.get("answer", "").strip()

        if not question or not answer:
            raise HTTPException(status_code=400, detail="Both question and answer are required.")

        embedding = compute_embedding(question).tobytes()
        conn = connect_db()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO ValidatedQA (question, answer, embedding) VALUES (?, ?, ?)",
            (question, answer, embedding),
        )
        conn.commit()
        conn.close()
        return {"message": "Question-Answer pair added successfully."}
    except sqlite3.Error as e:
        logging.error(f"Database error: {e}")
        raise HTTPException(status_code=500, detail="Database Error")

@app.get("/list-sections")
def list_sections():
    try:
        conn = connect_db()
        cursor = conn.cursor()
        cursor.execute("SELECT id, content FROM Sections LIMIT 10;")
        sections = cursor.fetchall()
        conn.close()

        return {"sections": [{"id": sec[0], "content": sec[1]} for sec in sections]}
    except sqlite3.Error as e:
        logging.error(f"Database error: {e}")
        raise HTTPException(status_code=500, detail="Database Error")

@app.post("/test-llama")
async def test_llama(prompt: str):
    try:
        inputs = llama_tokenizer(prompt, return_tensors="pt").to(device)
        outputs = llama_model.generate(
            **inputs,
            max_new_tokens=40,
            do_sample=True,
            top_k=20,
            temperature=0.7
        )
        response = llama_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return {"prompt": prompt, "response": response}
    except Exception as e:
        logging.error(f"Error in /test-llama endpoint: {e}")
        return {"error": str(e)}

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse("static/favicon.ico")
