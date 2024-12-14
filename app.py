from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import sqlite3
import numpy as np
from datetime import datetime, timedelta
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from rapidfuzz import fuzz, process
import json
import re
import logging
import os
import time
from apscheduler.schedulers.background import BackgroundScheduler
import uuid
from functools import lru_cache
from collections import defaultdict


SESSION_TIMEOUT = timedelta(hours=1)
# In-memory storage for session-based memory
session_memory = defaultdict(list)  # {session_id: [(query, response), ...]}
conversation_context = defaultdict(bool)  # Tracks if the session is EPR-related

# Global variables for dynamic keyword storage and frequency tracking
DYNAMIC_KEYWORDS = set()  # Set to store unique keywords
keyword_frequency = defaultdict(int)  # Defaultdict to track keyword frequency



# Define clean_expired_sessions before using it in the scheduler
def clean_expired_sessions():
    """Clean expired sessions based on the SESSION_TIMEOUT."""
    try:
        now = datetime.now()
        expired_sessions = []

        # Iterate through all sessions
        for session_id, interactions in list(session_memory.items()):
            if not interactions:
                continue
            # Check if the last interaction in the session is older than the timeout
            if now - interactions[-1]["timestamp"] > SESSION_TIMEOUT:
                expired_sessions.append(session_id)
                del session_memory[session_id]

        # Log cleaned sessions
        if expired_sessions:
            logger.info(f"Cleaned expired sessions: {expired_sessions}")
        else:
            logger.info("No expired sessions found during clean-up.")
    except Exception as e:
        logger.exception("Error during session clean-up")
        
scheduler = BackgroundScheduler()
scheduler.add_job(clean_expired_sessions, 'interval', hours=1)
scheduler.start()

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

# Define lifespan event handlers
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown."""
    logger.info("Application startup: Initializing resources.")
    load_keywords_from_file()  # Load keywords during startup
    yield
    save_keywords_to_file()  # Save keywords during shutdown
    logger.info("Application shutdown: Cleaning up resources.")

async def log_request_id(request: Request):
    request_id = str(uuid.uuid4())
    logger.info(f"Request ID: {request_id} - {request.method} {request.url}")
    return request_id

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
    if not os.path.exists(index_file):
        logger.error("index.html not found")
        raise HTTPException(status_code=404, detail="Frontend index.html not found")
    return FileResponse(index_file)

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
    """Load the Sentence-BERT model."""
    try:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Sentence-BERT model loaded successfully.")
        return model
    except Exception as e:
        logger.exception("Failed to load Sentence-BERT model")
        raise RuntimeError(f"Failed to load Sentence-BERT model: {e}")


llama_model = None
llama_tokenizer = None

def get_llama_model():
    global llama_model, llama_tokenizer
    if llama_model is None or llama_tokenizer is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Define the device
        llama_tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-2-7b-chat-hf",
            token=hf_token  # Replace deprecated `use_auth_token` with `token`
        )
        llama_model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-2-7b-chat-hf",
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True
        ).to(device)
        llama_model.eval()
    return llama_model, llama_tokenizer

def convert_to_native(value):
    """Convert numpy types to native Python types."""
    if isinstance(value, np.generic):
        return value.item()
    return value

# Suppress symlink warnings for Hugging Face cache (Windows-specific)
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="cache-system uses symlinks by default")


def preprocess_query(query):
    # Remove years and extra spaces
    query = re.sub(r'\b\d{4}\b', '', query)
    # Convert to lowercase and strip whitespace
    query = query.lower().strip()
    return query

def find_exact_or_partial_match(query, db_questions):
    for db_question in db_questions:
        if query in db_question.lower() or db_question.lower() in query:
            return db_question  # Return the closest match
    return None

def fuzzy_match_question(query, db_questions, threshold=75):
    match = process.extractOne(query, db_questions, scorer=fuzz.ratio)
    if match and match[1] >= threshold:
        return match[0]  # Return the best-matching question
    return None


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
        embedding = model.encode(text).reshape(1, -1)
        logger.info(f"Computed embedding for text: {text}")
        return embedding
    except Exception as e:
        logger.exception("Error computing embedding")
        raise


reference_queries = [
    "What is EPR?",
    "Explain plastic waste management.",
    "What are EPR compliance rules?",
    "How do I register for EPR compliance?"
]
reference_embeddings = np.vstack([compute_embedding(q) for q in reference_queries])


def is_query_relevant(query: str, reference_embeddings: np.ndarray, threshold: float = 0.7) -> bool:
    """Determine if a query is relevant to EPR using semantic similarity."""
    try:
        # Compute query embedding
        query_embedding = compute_embedding(query)
        # Compare with reference embeddings
        similarities = cosine_similarity(query_embedding, reference_embeddings)
        max_similarity = max(similarities[0])  # Get the highest similarity score
        logger.info(f"Max similarity for query '{query}': {max_similarity}")
        return max_similarity >= threshold
    except Exception as e:
        logger.exception("Error in is_query_relevant")
        raise
        

def load_keywords_from_file():
    """Load dynamic keywords from a file."""
    global DYNAMIC_KEYWORDS, keyword_frequency
    try:
        if os.path.exists("dynamic_keywords.json"):
            if os.path.getsize("dynamic_keywords.json") > 0:  # Check if file is not empty
                with open("dynamic_keywords.json", "r") as f:
                    data = json.load(f)
                    DYNAMIC_KEYWORDS.update(data.get("keywords", []))
                    keyword_frequency.update(data.get("frequency", {}))
                    logger.info("Keywords successfully loaded from file.")
            else:
                logger.warning("Keyword file is empty. Starting fresh.")
        else:
            logger.warning("Keyword file does not exist. Starting fresh.")
    except json.JSONDecodeError as e:
        logger.error(f"Keyword file is corrupted: {e}. Starting fresh.")
    except Exception as e:
        logger.error(f"Unexpected error loading keywords: {e}. Starting fresh.")
    finally:
        # Ensure keywords and frequency are initialized to avoid errors
        DYNAMIC_KEYWORDS = DYNAMIC_KEYWORDS or set()
        keyword_frequency = keyword_frequency or defaultdict(int)

def learn_keywords_from_query(question):
    global DYNAMIC_KEYWORDS, keyword_frequency
    # Extract keywords from the query (can be a simple split or NLP-based extraction)
    keywords = re.findall(r'\b\w+\b', question.lower())
    for keyword in keywords:
        DYNAMIC_KEYWORDS.add(keyword)
        keyword_frequency[keyword] += 1
    logger.info(f"Learned keywords: {keywords}")


def save_keywords_to_file():
    """Save dynamic keywords to a file."""
    try:
        with open("dynamic_keywords.json", "w") as f:
            json.dump(
                {"keywords": list(DYNAMIC_KEYWORDS), "frequency": dict(keyword_frequency)},
                f,
                indent=4
            )
        logger.info("Keywords successfully saved to file.")
    except Exception as e:
        logger.error(f"Failed to save keywords to file: {e}")


def query_validated_qa(user_embedding, question: str):
    """Query the ValidatedQA table for the best match."""
    try:
        conn = connect_db()
        cursor = conn.cursor()
        cursor.execute("SELECT question, answer, embedding FROM ValidatedQA")

        max_similarity = 0.0
        best_answer = None

        for row in cursor.fetchall():
            db_question, db_answer, db_embedding = row
            db_embedding_array = np.frombuffer(db_embedding, dtype=np.float32).reshape(1, -1)

            # Compute similarity
            similarity = cosine_similarity(user_embedding, db_embedding_array)[0][0]
            logger.debug(f"Similarity for '{question}' with '{db_question}': {similarity}")

            # Check for the best match
            if similarity > max_similarity:
                max_similarity = similarity
                best_answer = db_answer

        conn.close()

        if max_similarity >= 0.7:
            logger.info(f"Best match found with confidence {max_similarity}: {best_answer}")
            return best_answer, max_similarity

        logger.info(f"No relevant match found for query: {question}")
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
            logger.info(f"Fuzzy match attempt: '{match}' with score {score}")
            
            # Lower threshold for specific keywords
            threshold = 70 if "epr" in question.lower() else 80

            if score >= threshold:  # Adjust threshold dynamically
                logger.info(f"Fuzzy match accepted: {match} with score {score}")
                return FALLBACK_KB[match]
            else:
                logger.warning(f"Low confidence match: {match} with score {score}")
        else:
            logger.info("No suitable fuzzy match found")
        return None
    except Exception as e:
        logger.exception("Error during fuzzy matching")
        return None

def refine_with_llama(question: str, db_answer: str):
    try:
        llama_model, llama_tokenizer = get_llama_model()
        prompt = (
            "Rephrase this information to directly answer the question:\n\n"
            f"Question: {question}\n\n"
            f"Answer: {db_answer}\n\n"
            "Provide a concise and direct response:"
        )
        inputs = llama_tokenizer(prompt, return_tensors="pt").to(llama_model.device)
        outputs = llama_model.generate(
            **inputs,
            max_new_tokens=140,
            do_sample=True,
            top_k=50,
            temperature=0.7
        )
        refined_response = llama_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        return refined_response
    except Exception as e:
        logger.exception("Error refining response with LLaMA")
        return db_answer

def build_memory_context(session_id):
    """Build context from session memory."""
    if session_id not in session_memory or not session_memory[session_id]:
        return ""
    memory_context = " ".join(
        [f"User: {interaction['query']} Bot: {interaction['response']}" for interaction in session_memory[session_id]]
    )
    logger.debug(f"Memory context for session {session_id}: {memory_context}")
    return memory_context


# Custom Exception Handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.warning(f"HTTP error occurred: {exc.detail}")
    return JSONResponse(status_code=exc.status_code, content={"message": exc.detail})

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception occurred")
    return JSONResponse(status_code=500, content={"message": "An unexpected error occurred. Please try again later."})


@app.get("/health")
async def health_check():
    """Check the health of the application."""
    return {"status": "ok"}

# Chat Endpoint
@app.post("/chat")
async def chat_endpoint(request: Request):
    try:
        start_time = time.time()

        # Parse request data
        data = await request.json()
        question = preprocess_query(data.get("message", "").strip())
        session_id = data.get("session_id", "default")

        if not question:
            logger.warning("Received empty message")
            raise HTTPException(status_code=400, detail="Message cannot be empty.")

        # Log new session creation
        if session_id not in session_memory:
            logger.info(f"New session created: {session_id}")

        # Step 1: Add the current query to session memory
        session_memory[session_id].append({
            "query": question,
            "response": "Processing...",
            "timestamp": datetime.now()
        })

        # Limit memory size
        if len(session_memory[session_id]) > 5:
            session_memory[session_id].pop(0)

        # Load keywords dynamically
        load_keywords_from_file()

        # Step 2: Check for exact or partial match in the database
        conn = connect_db()
        cursor = conn.cursor()
        cursor.execute("SELECT question, answer FROM ValidatedQA")
        db_entries = cursor.fetchall()
        db_questions = [row[0] for row in db_entries]
        partial_match = find_exact_or_partial_match(question, db_questions)

        if partial_match:
            db_answer = next((answer for q, answer in db_entries if q == partial_match), None)
            logger.info(f"Exact or partial match found: {partial_match}")
            session_memory[session_id][-1]["response"] = db_answer
            return {
                "answer": db_answer,
                "confidence": 1.0,
                "source": "exact or partial match",
                "response_time": f"{time.time() - start_time:.2f} seconds",
            }

        # Step 3: Check query relevance using embeddings
        memory_context = build_memory_context(session_id)
        full_query = f"{memory_context} User: {question}" if memory_context else question
        user_embedding = compute_embedding(full_query)

        db_answer, confidence = query_validated_qa(user_embedding, question)

        if db_answer and confidence >= 0.7:  # Lowered threshold
            logger.info(f"Database response found for query: {question} with confidence {confidence}")
            refined_answer = refine_with_llama(question, db_answer)
            session_memory[session_id][-1]["response"] = refined_answer
            learn_keywords_from_query(question)
            return {
                "answer": refined_answer,
                "confidence": convert_to_native(confidence),
                "source": "database + llama refinement",
                "response_time": f"{time.time() - start_time:.2f} seconds",
            }

        # Step 4: Fallback for no relevant database match
        fallback_response = fuzzy_match_fallback(question)
        session_memory[session_id][-1]["response"] = fallback_response or "No relevant information found."
        learn_keywords_from_query(question)
        return {
            "answer": fallback_response or "I couldn't find relevant information.",
            "confidence": 0.5,
            "source": "fuzzy fallback",
            "response_time": f"{time.time() - start_time:.2f} seconds",
        }

     except HTTPException as e:
        logger.warning(f"HTTP error: {e.detail}")
        raise e
     except TypeError as te:
        logger.error(f"Serialization error: {te}")
        raise HTTPException(status_code=500, detail="Response serialization error.")
     except Exception as e:
        logger.exception("Unhandled error in /chat endpoint")
        raise HTTPException(status_code=500, detail="An internal server error occurred.")
