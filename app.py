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
import logging
import os
import random
import time

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
    "how do you work": "I analyze your questions, look up answers in a database, and refine them using an advanced AI model for conversational responses."
}

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Define lifespan event handlers
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown."""
    logging.info("Application startup: Initializing resources.")
    yield
    logging.info("Application shutdown: Cleaning up resources.")

# Initialize FastAPI app with lifespan
app = FastAPI(lifespan=lifespan)

# Directory paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_FILES_DIR = os.path.join(CURRENT_DIR, "static")
TEMPLATES_DIR = os.path.join(CURRENT_DIR, "templates")
DB_PATH = os.path.join(CURRENT_DIR, "knowledge_base.db")

# Mount static files
app.mount("/static", StaticFiles(directory=STATIC_FILES_DIR), name="static")

# Hugging Face token
hf_token = "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"

# Serve `index.html` for root route
@app.get("/")
async def read_root():
    """Serve the index.html file."""
    index_file = os.path.join(TEMPLATES_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    raise HTTPException(status_code=404, detail="Frontend index.html not found")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Sentence-BERT model
try:
    model = SentenceTransformer("all-MiniLM-L6-v2")
    logging.info("Sentence-BERT model loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load Sentence-BERT model: {e}")
    raise RuntimeError(f"Failed to load Sentence-BERT model: {e}")

# Adjust LLaMA 2 model loading
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

# Suppress symlink warnings for Hugging Face cache (Windows-specific)
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="cache-system uses symlinks by default")

# Utility functions
def connect_db():
    """Connect to the SQLite database."""
    try:
        return sqlite3.connect(DB_PATH)
    except sqlite3.Error as e:
        logging.error(f"Database connection error: {e}")
        raise HTTPException(status_code=500, detail="Database connection failed.")

def compute_embedding(text: str):
    """Compute embedding for a given text using Sentence-BERT."""
    return model.encode(text).reshape(1, -1)

def query_validated_qa(user_embedding):
    """Query the ValidatedQA table for the best match."""
    try:
        conn = connect_db()
        cursor = conn.cursor()
        cursor.execute("SELECT question, answer, embedding FROM ValidatedQA")
        rows = cursor.fetchall()

        max_similarity = 0.0
        best_answer = None

        for _, db_answer, db_embedding in rows:
            db_embedding_array = np.frombuffer(db_embedding, dtype=np.float32).reshape(1, -1)
            similarity = cosine_similarity(user_embedding, db_embedding_array)[0][0]
            if similarity > max_similarity:
                max_similarity = similarity
                best_answer = db_answer

        conn.close()
        if max_similarity >= 0.7:  # Similarity threshold
            return best_answer, float(max_similarity)
        return None, 0.0
    except sqlite3.Error as e:
        logging.error(f"Database query error: {e}")
        return None, 0.0

def search_sections(query: str):
    """Search for terms in the sections table."""
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
    """Determine a dynamic opening based on the query type."""
    query = query.lower()
    if any(keyword in query for keyword in ["when", "date", "time", "timeline"]):
        return random.choice(OPENINGS["time"])
    elif any(keyword in query for keyword in ["who", "what", "fact"]):
        return random.choice(OPENINGS["fact"])
    else:
        return random.choice(OPENINGS["default"])

# Chat Endpoint with Fallback Behavior
@app.post("/chat")
async def chat_endpoint(request: Request):
    try:
        # Start timing
        start_time = time.time()

        # Parse the user query
        data = await request.json()
        question = data.get("message", "").strip().lower()

        if not question:
            raise HTTPException(status_code=400, detail="Message cannot be empty.")

        # Handle predefined fallback queries
        if question in FALLBACK_KB:
            response_time = time.time() - start_time
            return {
                "answer": FALLBACK_KB[question],
                "confidence": 1.0,
                "source": "fallback knowledge base",
                "response_time": f"{response_time:.2f} seconds",
            }

        # Step 1: Compute embedding for the question
        user_embedding = compute_embedding(question)

        # Step 2: Query the database for a relevant answer
        answer, confidence = query_validated_qa(user_embedding)

        # Step 3: Use LLaMA to refine the response if a valid database match is found
        if answer and confidence >= 0.8:
            # Construct the rephrasing prompt
            prompt = f"Rephrase this information in a friendly and conversational tone:\n\n{answer}"

            # Tokenize and process with LLaMA
            inputs = llama_tokenizer(prompt, return_tensors="pt").to(llama_model.device)
            outputs = llama_model.generate(
                **inputs,
                max_new_tokens=120,
                do_sample=True,
                top_k=40,
                temperature=0.8
            )
            refined_response = llama_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

            # Post-process to clean up the output
            final_answer = refined_response.split("\n\n")[-1].strip()

            # Calculate response time
            response_time = time.time() - start_time

            # Return the refined response with response time
            return {
                "answer": final_answer,
                "confidence": confidence,
                "source": "database + llama",
                "response_time": f"{response_time:.2f} seconds",
            }

        # Step 4: Handle cases where no valid answer is found in database or fallback KB
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
    """Add a new question-answer pair to the database."""
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
    """List all sections available in the database."""
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
    """Test the LLaMA 2 model with a given prompt."""
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
