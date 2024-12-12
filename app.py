from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
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
from collections import defaultdict


# In-memory storage for session-based memory
session_memory = defaultdict(list)  # {session_id: [(query, response), ...]}
conversation_context = defaultdict(bool)  # Tracks if the session is EPR-related

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app_log.log"),  # Log to a file
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
    "what can you do": "I can answer questions about Extended Producer Responsibility (EPR) and assist with understanding concepts like plastic waste management, rules, and responsibilities. Try asking something specific!",
    "who made you": "I was developed as a collaborative effort to assist with Extended Producer Responsibility (EPR) and related topics using advanced AI capabilities!",
    "how do you work": "I analyze your questions, look up answers in a database, and refine them using an advanced AI model for conversational responses related to Extended Producer Responsibility."
}

# EPR Keywords for Validation
EPR_KEYWORDS = [
    "epr",
    "extended producer responsibility",
    "plastic waste management",
    "recycling",
    "environment",
    "producer responsibility",
    "waste",
    "management",
    "sustainability",
    "eco-friendly",
    "pollution control",
    "waste collection",
    "waste segregation",
    "recyclable materials",
    "plastic recycling",
    "epr compliance",
    "environmental responsibility",
    "post-consumer waste",
    "material recovery",
    "circular economy",
    "waste reduction",
    "responsible manufacturing",
    "waste disposal",
    "eco-conscious",
    "epr targets",
    "green initiatives",
    "sustainable packaging",
    "producer registration",
    "epr schemes",
    "end-of-life products",
    "collection and recycling",
    "resource efficiency",
    "environmental impact",
    "epr rules",
    "plastic credits",
    "recovery obligations",
    "compliance reporting",
    "sustainable development"
]

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
        return sqlite3.connect(DB_PATH)
    except sqlite3.Error as e:
        logger.error(f"Database connection error: {e}")
        raise HTTPException(status_code=500, detail="Database connection failed.")

def compute_embedding(text: str):
    """Compute embedding for a given text using Sentence-BERT."""
    try:
        model = load_sentence_bert()
        return model.encode(text).reshape(1, -1)
    except Exception as e:
        logger.exception("Error computing embedding")
        raise

def is_query_relevant(query: str) -> bool:
    """Check if the query is relevant to Extended Producer Responsibility."""
    query = query.lower()
    return any(keyword in query for keyword in EPR_KEYWORDS)

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

            _, db_answer, db_embedding = row
            db_embedding_array = np.frombuffer(db_embedding, dtype=np.float32).reshape(1, -1)
            similarity = cosine_similarity(user_embedding, db_embedding_array)[0][0]

            if similarity > max_similarity:
                max_similarity = similarity
                best_answer = db_answer

        conn.close()

        if max_similarity >= 0.8:  # Similarity threshold
            return best_answer, float(max_similarity)

        return None, 0.0
    except sqlite3.Error as e:
        logger.error(f"Database query error: {e}")
        return None, 0.0

def fuzzy_match_fallback(question: str) -> str:
    """Use rapidfuzz to find the closest fallback response."""
    try:
        # Extract the closest match using RapidFuzz
        result = process.extractOne(question, FALLBACK_KB.keys(), scorer=fuzz.ratio)
        if result:
            match, score = result[0], result[1]
            if score >= 80:  # Set threshold for acceptable match
                logger.info(f"Fuzzy match found: {match} with score {score}")
                return FALLBACK_KB[match]
            else:
                logger.warning(f"Low confidence match: {match} with score {score}")
        else:
            logger.info("No suitable fuzzy match found")
        return None
    except Exception as e:
        logger.exception("Error during fuzzy matching")
        return None


# Custom Exception Handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.warning(f"HTTP error occurred: {exc.detail}")
    return JSONResponse(status_code=exc.status_code, content={"message": exc.detail})

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception occurred")
    return JSONResponse(status_code=500, content={"message": "An unexpected error occurred. Please try again later."})

# Chat Endpoint
@app.post("/chat")
async def chat_endpoint(request: Request):
    try:
        start_time = time.time()

        # Parse the user query and session_id
        data = await request.json()
        question = data.get("message", "").strip().lower()
        session_id = data.get("session_id", "default")

        if not question:
            logger.warning("Received empty message")
            raise HTTPException(status_code=400, detail="Message cannot be empty.")

        # Step 1: Check if the conversation is already EPR-related
        is_context_epr_related = conversation_context[session_id]

        # Step 2: Validate the query relevance to EPR (only for initial questions)
        if not is_context_epr_related and not is_query_relevant(question):
            logger.info("Rejected irrelevant query: %s", question)
            return {
                "answer": "I can only assist with questions related to Extended Producer Responsibility (EPR).",
                "confidence": 0.0,
                "source": "query validation",
                "response_time": f"{time.time() - start_time:.2f} seconds",
            }

        # Step 3: Update context as EPR-related if the initial query is relevant
        if not is_context_epr_related:
            conversation_context[session_id] = True

        # Step 4: Build a dynamic prompt using conversation history
        memory_context = " ".join([f"User: {q} Bot: {r}" for q, r in session_memory[session_id]])
        full_query = f"{memory_context} User: {question}" if memory_context else question

        # Step 5: Compute embedding for the full query
        user_embedding = compute_embedding(full_query)

        # Step 6: Query the database for a relevant answer
        answer, confidence = query_validated_qa(user_embedding)
        confidence = float(confidence)

        # Step 7: Handle fallback responses
        fallback_response = fuzzy_match_fallback(question)
        if fallback_response:
            session_memory[session_id].append((question, fallback_response))
            if len(session_memory[session_id]) > 5:
                session_memory[session_id].pop(0)
            return {
                "answer": fallback_response,
                "confidence": 1.0,
                "source": "fuzzy fallback knowledge base",
                "response_time": f"{time.time() - start_time:.2f} seconds",
            }

        
        if answer and confidence >= 0.8:
            # Use LLaMA to refine the response
            prompt = f"Rephrase this information in a professional and clear tone:\n\n{answer}"
            inputs = llama_tokenizer(prompt, return_tensors="pt").to(llama_model.device)
            outputs = llama_model.generate(
                **inputs,
                max_new_tokens=140,
                do_sample=True,
                top_k=50,
                temperature=0.7
            )
            refined_response = llama_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

            final_answer = refined_response.split("\n\n")[-1].strip()

            # Update memory with the latest interaction
            session_memory[session_id].append((question, final_answer))
            if len(session_memory[session_id]) > 5:  # Limit memory size
                session_memory[session_id].pop(0)

            return {
                "answer": final_answer,
                "confidence": confidence,
                "source": "database + llama",
                "response_time": f"{time.time() - start_time:.2f} seconds",
            }

        # Step 7: Handle fallback responses
        fallback_response = fuzzy_match_fallback(question)
        if fallback_response:
            session_memory[session_id].append((question, fallback_response))
            if len(session_memory[session_id]) > 5:
                session_memory[session_id].pop(0)
            return {
                "answer": fallback_response,
                "confidence": 1.0,
                "source": "fuzzy fallback knowledge base",
                "response_time": f"{time.time() - start_time:.2f} seconds",
            }

        # Step 8: Default response for no matches
        fallback_answer = "I'm sorry, I couldn't find relevant information."
        session_memory[session_id].append((question, fallback_answer))
        if len(session_memory[session_id]) > 5:
            session_memory[session_id].pop(0)
        return {
            "answer": fallback_answer,
            "confidence": 0.0,
            "source": "fallback response",
            "response_time": f"{time.time() - start_time:.2f} seconds",
        }

    except HTTPException as e:
        raise e  # Let custom handler handle HTTP errors
    except Exception as e:
        logger.exception("Error in /chat endpoint")
        raise HTTPException(status_code=500, detail="An internal server error occurred.")
