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

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Define lifespan event handlers
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown."""
    logging.info("Application startup: Initializing resources.")
    # Add startup tasks here (e.g., model loading, database connections)
    yield
    logging.info("Application shutdown: Cleaning up resources.")
    # Add cleanup tasks here (e.g., closing database connections)

# Initialize FastAPI app with lifespan
app = FastAPI(lifespan=lifespan)

# Hardcoded database path
DB_PATH = r"D:\EPR Data\Updated db'\knowledge_base.db"

# Hugging Face token
hf_token = "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"

# Define the directory for static files
STATIC_FILES_DIR = os.path.dirname(os.path.abspath(__file__))

# Mount static files
app.mount("/static", StaticFiles(directory=STATIC_FILES_DIR), name="static")

# Serve index.html for root route
@app.get("/")
async def read_root():
    """Serve the index.html file."""
    index_file = os.path.join(STATIC_FILES_DIR, "index.html")
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
    raise RuntimeError(f"Failed to load Sentence-BERT model: {e}")

# Load Hugging Face LLaMA 2 model
try:
    llama_tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf",  # Use a valid model ID
    token=hf_token
    )
    llama_model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-chat-hf",
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        token=hf_token
)

    llama_model.eval()  # Set model to evaluation mode

    logging.info("LLaMA 2 model loaded successfully.")
except Exception as e:
    raise RuntimeError(f"Failed to load LLaMA 2 model: {e}")

# Suppress symlink warnings for Hugging Face cache (Windows-specific)
import warnings
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="cache-system uses symlinks by default"
)

# Utility functions
def connect_db():
    """Connect to the SQLite database."""
    try:
        return sqlite3.connect(DB_PATH)
    except sqlite3.Error as e:
        logging.error(f"Database connection error: {e}")
        raise HTTPException(status_code=500, detail="Database connection failed.")

def compute_embedding(text):
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

@app.post("/chat")
async def chat_endpoint(request: Request):
    """
    Handle user queries by fetching answers from the database
    and rephrasing them with LLaMA for conversational flow.
    If no answer is found in the database, return a polite, non-speculative message.
    """
    try:
        # Parse user input
        data = await request.json()
        question = data.get("message", "").strip()

        if not question:
            raise HTTPException(status_code=400, detail="Message cannot be empty.")

        # Step 1: Fetch Answer from Database
        user_embedding = compute_embedding(question)
        answer, confidence = query_validated_qa(user_embedding)

        # Step 2: If answer is found, rephrase using LLaMA
        if answer:
            # Create a prompt for LLaMA
            prompt = f"""
            You are a helpful assistant engaged in a conversation. Rephrase the following factual information to match the conversational tone of the current discussion:

            Database Answer: "{answer}"

            User Question: "{question}"
            """
            try:
                # Generate a conversational response using LLaMA
                inputs = llama_tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
                outputs = llama_model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=True,
                    top_k=50,
                    temperature=0.7
                )
                enriched_response = llama_tokenizer.decode(outputs[0], skip_special_tokens=True)

                return {
                    "answer": enriched_response,
                    "confidence": confidence,
                    "source": "database + llama",
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
            except Exception as llama_error:
                logging.error(f"LLaMA rephrasing error: {llama_error}")
                return {
                    "answer": "An error occurred while generating a conversational response. Please try again.",
                    "confidence": confidence,
                    "source": "database",
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }

        # Step 3: No Database Match
        return {
            "answer": "I'm sorry, I couldn't find any relevant information in the database.",
            "confidence": 0.0,
            "source": "database",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    except Exception as e:
        logging.error(f"Error in /chat endpoint: {e}")
        return {
            "answer": "An internal error occurred. Please try again later.",
            "confidence": 0.0,
            "source": "error",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
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
        # Tokenize the input prompt
        inputs = llama_tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")

        # Generate response
        outputs = llama_model.generate(
        **inputs,
        max_new_tokens=40,  # Reduce token generation limit
        do_sample=True,
        top_k=20,           # Limit sampling diversity for smaller models
        temperature=0.7     # Balance creativity with relevance
    )


        # Decode and return the response
        response = llama_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return {"prompt": prompt, "response": response}
    except Exception as e:
        return {"error": str(e)}
