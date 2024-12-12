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
from fuzzywuzzy import fuzz, process
import logging
import os
import random
import time
from functools import lru_cache

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app_debug.log"),  # Log to a file
        logging.StreamHandler()  # Log to the console
    ]
)
logger = logging.getLogger("EPR_Chatbot")

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

# Define lifespan event handlers
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown."""
    logger.info("Application startup: Initializing resources.")
    yield
    logger.info("Application shutdown: Cleaning up resources.")

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
        logger.debug("Serving index.html")
        return FileResponse(index_file)
    logger.error("index.html not found")
    raise HTTPException(status_code=404, detail="Frontend index.html not found")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Sentence-BERT model with caching
@lru_cache(maxsize=1)
def load_sentence_bert():
    try:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Sentence-BERT model loaded successfully.")
        return model
    except Exception as e:
        logger.exception("Failed to load Sentence-BERT model")
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
    logger.info("LLaMA 2 model loaded successfully on %s.", device)
except Exception as e:
    logger.exception("Failed to load LLaMA 2 model")
    raise RuntimeError(f"Failed to load LLaMA 2 model: {e}")

# Suppress symlink warnings for Hugging Face cache (Windows-specific)
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="cache-system uses symlinks by default")

# Utility functions
def connect_db():
    """Connect to the SQLite database."""
    try:
        logger.debug("Connecting to the database.")
        return sqlite3.connect(DB_PATH)
    except sqlite3.Error as e:
        logger.error(f"Database connection error: {e}")
        raise HTTPException(status_code=500, detail="Database connection failed.")

def compute_embedding(text: str):
    """Compute embedding for a given text using Sentence-BERT."""
    try:
        logger.debug(f"Computing embedding for text: {text}")
        model = load_sentence_bert()
        embedding = model.encode(text).reshape(1, -1)
        logger.debug("Embedding computed successfully.")
        return embedding
    except Exception as e:
        logger.exception("Error computing embedding")
        raise

def query_validated_qa(user_embedding):
    """Query the ValidatedQA table for the best match."""
    try:
        conn = connect_db()
        cursor = conn.cursor()
        cursor.execute("SELECT question, answer, embedding FROM ValidatedQA")
        max_similarity = 0.0
        best_answer = None

        while True:
            row = cursor.fetchone()
            if row is None:
                break

            db_question, db_answer, db_embedding = row
            db_embedding_array = np.frombuffer(db_embedding, dtype=np.float32).reshape(1, -1)
            similarity = cosine_similarity(user_embedding, db_embedding_array)[0][0]

            logger.debug(f"Processed row with similarity {similarity} for question: {db_question}")

            if similarity > max_similarity:
                max_similarity = similarity
                best_answer = db_answer

        conn.close()

        if max_similarity >= 0.7:  # Similarity threshold
            logger.info(f"Best match found: {best_answer} with similarity {max_similarity}")
            return best_answer, max_similarity

        logger.info("No suitable match found.")
        return None, 0.0
    except sqlite3.Error as e:
        logger.error(f"Database query error: {e}")
        return None, 0.0

def fuzzy_match_fallback(question: str) -> str:
    """Use fuzzy matching to find the closest fallback response."""
    try:
        logger.debug(f"Fuzzy matching for question: {question}")
        match, score = process.extractOne(question, FALLBACK_KB.keys(), scorer=fuzz.ratio)
        if score >= 80:  # Set threshold for acceptable match
            logger.info(f"Fuzzy match found: {match} with score {score}")
            return FALLBACK_KB[match]
        logger.warning(f"No close match found for question: '{question}'")
        return None
    except Exception as e:
        logger.exception("Error during fuzzy matching")
        return None

def search_sections(query: str):
    """Search for terms in the sections table."""
    try:
        logger.debug(f"Searching sections for query: {query}")
        conn = connect_db()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, content FROM Sections WHERE content LIKE ? LIMIT 10;",
            (f"%{query}%",),
        )
        results = cursor.fetchall()
        conn.close()
        logger.info(f"Found {len(results)} matching sections.")
        return [{"id": row[0], "content": row[1]} for row in results]
    except sqlite3.Error as e:
        logger.error(f"Database error: {e}")
        return []

def get_dynamic_opening(query: str) -> str:
    """Determine a dynamic opening based on the query type."""
    try:
        logger.debug(f"Determining dynamic opening for query: {query}")
        query = query.lower()
        if any(keyword in query for keyword in ["when", "date", "time", "timeline"]):
            opening = random.choice(OPENINGS["time"])
        elif any(keyword in query for keyword in ["who", "what", "fact"]):
            opening = random.choice(OPENINGS["fact"])
        else:
            opening = random.choice(OPENINGS["default"])
        logger.info(f"Dynamic opening selected: {opening}")
        return opening
    except Exception as e:
        logger.exception("Error determining dynamic opening")
        return ""

# Chat Endpoint with Fallback Behavior
@app.post("/chat")
async def chat_endpoint(request: Request):
    try:
        # Start timing
        start_time = time.time()

        # Parse the user query
        data = await request.json()
        question = data.get("message", "").strip().lower()
        logger.info(f"Received question: {question}")

        if not question:
            logger.warning("Received empty message")
            raise HTTPException(status_code=400, detail="Message cannot be empty.")

        # Handle predefined fallback queries with fuzzy matching
        fallback_response = fuzzy_match_fallback(question)
        if fallback_response:
            response_time = time.time() - start_time
            logger.info(f"Fallback response: {fallback_response}")
            return {
                "answer": fallback_response,
                "confidence": 1.0,
                "source": "fuzzy fallback knowledge base",
                "response_time": f"{response_time:.2f} seconds",
            }

        # Step 1: Compute embedding for the question
        user_embedding = compute_embedding(question)

        # Step 2: Query the database for a relevant answer
        answer, confidence = query_validated_qa(user_embedding)

        # Step 3: Use LLaMA to refine the response if a valid database match is found
        if answer and confidence >= 0.8:
            prompt = f"Rephrase this information in a friendly and conversational tone:\n\n{answer}"
            logger.debug(f"Using LLaMA with prompt: {prompt}")

            inputs = llama_tokenizer(prompt, return_tensors="pt").to(llama_model.device)
            outputs = llama_model.generate(
                **inputs,
                max_new_tokens=140,
                do_sample=True,
                top_k=50,
                temperature=0.7
            )
            refined_response = llama_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

            # Post-process to clean up the output
            final_answer = refined_response.split("\n\n")[-1].strip()

            # Calculate response time
            response_time = time.time() - start_time

            logger.info(f"Final response: {final_answer}, confidence: {confidence}")
            return {
                "answer": final_answer,
                "confidence": confidence,
                "source": "database + llama",
                "response_time": f"{response_time:.2f} seconds",
            }

        # Step 4: Handle cases where no valid answer is found in database or fallback KB
        response_time = time.time() - start_time
        logger.info("No valid answer found. Returning fallback response.")
        return {
            "answer": "I'm sorry, I couldn't find relevant information. Feel free to ask about EPR or related topics!",
            "confidence": 0.0,
            "source": "fallback response",
            "response_time": f"{response_time:.2f} seconds",
        }

    except Exception as e:
        logger.exception("Error in /chat endpoint")
        response_time = time.time() - start_time
        return {
            "answer": "An internal error occurred. Please try again later.",
            "confidence": 0.0,
            "source": "error",
            "response_time": f"{response_time:.2f} seconds",
        }
