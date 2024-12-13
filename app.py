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
import json
import re
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

# Define lifespan event handlers
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown."""
    logger.info("Application startup: Initializing resources.")
    load_keywords_from_file()  # Load keywords during startup
    yield
    save_keywords_to_file()  # Save keywords during shutdown
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
        logger.info(f"Computed embedding shape: {embedding.shape}")
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



def load_keywords_from_file():
    """Load dynamic keywords from a file."""
    global DYNAMIC_KEYWORDS, keyword_frequency
    try:
        with open("dynamic_keywords.json", "r") as f:
            data = json.load(f)
            DYNAMIC_KEYWORDS.update(data.get("keywords", []))
            keyword_frequency.update(data.get("frequency", {}))
    except FileNotFoundError:
        logger.warning("Keyword file not found. Starting fresh.")


def query_validated_qa(user_embedding, question: str):
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

            # Compute similarity
            similarity = cosine_similarity(user_embedding, db_embedding_array)[0][0]

            # Log similarity for debugging
            logger.info(f"Similarity for query '{question}' with DB question '{db_question}': {similarity}")

            # Exact text match fallback
            if db_question.strip().lower() == question.strip().lower():
                logger.info(f"Exact match found for '{question}' in database.")
                return db_answer, 1.0

            # Track the best match
            if similarity > max_similarity:
                max_similarity = similarity
                best_answer = db_answer

        conn.close()

        # Return best match if similarity exceeds threshold
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
    """Refine database answer using LLaMA."""
    try:
        prompt = f"Rephrase this information to directly answer the question:\n\nQuestion: {question}\n\nAnswer: {db_answer}"
        inputs = llama_tokenizer(prompt, return_tensors="pt").to(llama_model.device)
        outputs = llama_model.generate(
            **inputs,
            max_new_tokens=140,
            do_sample=True,
            top_k=50,
            temperature=0.7
        )
        refined_response = llama_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        refined_answer = refined_response.split("\n\n")[-1].strip()
        logger.info(f"Refined response: {refined_answer}")
        return refined_answer
    except Exception as e:
        logger.exception("Error refining response with LLaMA")
        return db_answer  # Fallback to the original answer if LLaMA fails


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

        # Parse request data
        data = await request.json()
        question = preprocess_query(data.get("message", "").strip())  # Preprocess query
        session_id = data.get("session_id", "default")

        if not question:
            logger.warning("Received empty message")
            raise HTTPException(status_code=400, detail="Message cannot be empty.")

        # Load keywords dynamically (if applicable)
        load_keywords_from_file()

        # Step 1: Check for exact or partial match
        conn = connect_db()
        cursor = conn.cursor()
        cursor.execute("SELECT question FROM ValidatedQA")
        db_questions = [row[0] for row in cursor.fetchall()]
        partial_match = find_exact_or_partial_match(question, db_questions)

        if partial_match:
            cursor.execute("SELECT answer FROM ValidatedQA WHERE question = ?", (partial_match,))
            db_answer = cursor.fetchone()[0]
            logger.info(f"Exact or partial match found: {partial_match}")
            return {
                "answer": db_answer,
                "confidence": 1.0,
                "source": "exact or partial match",
                "response_time": f"{time.time() - start_time:.2f} seconds",
            }

        # Step 2: Check query relevance
        if not is_query_relevant(question, reference_embeddings):
            logger.info(f"Ambiguous or irrelevant query: {question}")
            # Attempt to process query anyway using fuzzy match fallback
            fallback_response = fuzzy_match_fallback(question)
            if fallback_response:
                return {
                    "answer": fallback_response,
                    "confidence": 0.5,
                    "source": "fallback for ambiguous query",
                    "response_time": f"{time.time() - start_time:.2f} seconds",
                }

        # Step 3: Use memory context for better embeddings
        memory_context = " ".join([f"User: {q} Bot: {r}" for q, r in session_memory[session_id]])
        full_query = f"{memory_context} User: {question}" if memory_context else question
        logger.info(f"Dynamic prompt for query: {full_query}")

        # Compute embedding for the full query
        user_embedding = compute_embedding(full_query)

        # Step 4: Query the database for a relevant answer
        db_answer, confidence = query_validated_qa(user_embedding, question)

        if db_answer and confidence >= 0.8:
            logger.info(f"Database response found for query: {question} with confidence {confidence}")
            refined_answer = refine_with_llama(question, db_answer)
            session_memory[session_id].append((question, refined_answer))
            if len(session_memory[session_id]) > 5:  # Limit memory size
                session_memory[session_id].pop(0)

            # Learn keywords dynamically
            learn_keywords_from_query(question)

            return {
                "answer": refined_answer,
                "confidence": float(confidence),
                "source": "database + llama refinement",
                "response_time": f"{time.time() - start_time:.2f} seconds",
            }

        # Step 5: Handle fallback with improved matching
        fallback_response = fuzzy_match_fallback(question)
        if fallback_response:
            logger.info(f"Fallback response used for query: {question}")
            session_memory[session_id].append((question, fallback_response))
            if len(session_memory[session_id]) > 5:
                session_memory[session_id].pop(0)

            # Learn keywords dynamically
            learn_keywords_from_query(question)

            return {
                "answer": fallback_response,
                "confidence": 1.0,
                "source": "fuzzy fallback",
                "response_time": f"{time.time() - start_time:.2f} seconds",
            }

        # Step 6: Default response for no matches
        logger.info(f"No valid response found for query: {question}")
        default_response = "I couldn't find relevant information. Please ask about Extended Producer Responsibility (EPR)."
        session_memory[session_id].append((question, default_response))
        if len(session_memory[session_id]) > 5:
            session_memory[session_id].pop(0)

        # Learn keywords dynamically
        learn_keywords_from_query(question)

        return {
            "answer": default_response,
            "confidence": 0.0,
            "source": "default fallback",
            "response_time": f"{time.time() - start_time:.2f} seconds",
        }

    except HTTPException as e:
        raise e
    except Exception as e:
        logger.exception("Error in /chat endpoint")
        raise HTTPException(status_code=500, detail="An internal server error occurred.")
