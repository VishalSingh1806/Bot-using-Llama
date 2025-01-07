# Standard Library Imports
import os
import re
import json
import time
import logging
import sqlite3  
from sqlite3 import Connection
from datetime import datetime, timedelta
from functools import lru_cache
from collections import defaultdict
from queue import Queue
from threading import Lock
import uuid
import torch
import numpy as np
import asyncio 
from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from spacy import load as spacy_load
from rapidfuzz import fuzz, process
from accelerate import infer_auto_device_map, init_empty_weights
from transformers import AutoTokenizer, AutoModelForCausalLM
from apscheduler.schedulers.background import BackgroundScheduler
import redis
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from prometheus_client import Summary, Counter, start_http_server, generate_latest, CONTENT_TYPE_LATEST
from logging.handlers import RotatingFileHandler
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


SESSION_TIMEOUT = timedelta(minutes=1)

# Load spaCy model globally during startup
nlp = spacy_load("en_core_web_sm")

# Modify the session memory structure
session_memory = defaultdict(lambda: {"history": [], "context": ""})  # Session structure
conversation_context = defaultdict(bool)  # Tracks if the session is EPR-related

# Global variables for dynamic keyword storage and frequency tracking
DYNAMIC_KEYWORDS = set()  # Set to store unique keywords
keyword_frequency = defaultdict(int)  # Defaultdict to track keyword frequency

# Global cache for storing recently processed questions and answers
CACHE = {}
CACHE_THRESHOLD = 0.9  # Minimum similarity for cache retrieval

# Cache for dynamic query embeddings
embedding_cache = {}


# Define clean_expired_sessions before using it in the scheduler
def clean_expired_sessions():
    """Clean expired sessions using Redis TTL."""
    try:
        keys = redis_client.keys("session:*")  # Fetch all session keys
        for key in keys:
            ttl = redis_client.ttl(key)  # Get time-to-live for the key
            if ttl == -2:  # Key does not exist (expired)
                redis_client.delete(key)
        logger.info("Expired sessions cleaned up successfully.")
    except Exception as e:
        logger.exception("Error during session cleanup.")

        
scheduler = BackgroundScheduler()
scheduler.add_job(clean_expired_sessions, 'interval', hours=1)
scheduler.start()


# Prometheus metrics
REQUEST_LATENCY = Summary('request_latency_seconds', 'Latency of HTTP requests')
ERROR_COUNT = Counter('error_count', 'Total number of errors')

# Start Prometheus server
start_http_server(9100)  # Exposes metrics on http://localhost:9100


# Logging configuration with rotation
LOG_LEVEL = logging.INFO
logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler("app_log.log", maxBytes=10 * 1024 * 1024, backupCount=5),  # 10 MB per file, 5 backups
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("EPR_Chatbot")

# Redis configuration
REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 0

# Initialize Redis client
try:
    redis_client = redis.StrictRedis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)
    redis_client.ping()
    logger.info("Connected to Redis successfully.")
except redis.ConnectionError as e:
    logger.error("Failed to connect to Redis.")
    raise RuntimeError("Redis connection failed.") from e

# Global session memory


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
    global llama_model, llama_tokenizer, reference_embeddings

    logger.info("Application startup: Initializing resources.")

    # Test database connection and structure
    logger.info("Testing database connection...")
    test_db_connection()  # Call the test function here

    # Load Sentence-BERT model (cached)
    load_sentence_bert()

    # Precompute static embeddings for reference queries
    try:
        reference_queries = [
            "What is EPR?",
            "Explain plastic waste management.",
            "What are EPR compliance rules?",
            "How do I register for EPR compliance?"
        ]
        reference_embeddings = np.vstack([compute_embedding(q) for q in reference_queries])
        logger.info("Static embeddings precomputed for reference queries.")
    except Exception as e:
        logger.error(f"Failed to precompute reference embeddings: {e}")

    # Load LLaMA model and tokenizer
    try:
        llama_tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-2-7b-chat-hf",
            token=hf_token
        )
        llama_model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-2-7b-chat-hf",
            device_map="auto",  # Automatically distribute across available GPUs
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        llama_model.eval()
        logger.info("LLaMA model and tokenizer loaded successfully during startup.")
    except Exception as e:
        logger.error(f"Failed to load LLaMA model and tokenizer during startup: {e}")
        raise RuntimeError("Model loading failed during startup.")

    load_keywords_from_file()  # Load keywords during startup
    yield
    save_keywords_to_file()  # Save keywords during shutdown
    logger.info("Application shutdown: Cleaning up resources.")

# Initialize FastAPI app with lifespan
app = FastAPI(lifespan=lifespan)

# Route to expose Prometheus metrics via FastAPI
@app.get("/metrics")
async def metrics():
    """Expose Prometheus metrics via FastAPI."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
    

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


def convert_to_native(value):
    """Convert numpy types to native Python types."""
    if isinstance(value, np.generic):
        return value.item()
    return value

# Suppress symlink warnings for Hugging Face cache (Windows-specific)
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="cache-system uses symlinks by default")


def preprocess_query(query, session_id):
    """Preprocess user query and resolve ambiguous references using session context."""
    query = re.sub(r'\b\d{4}\b', '', query)  # Remove years
    query = query.lower().strip()

    # Resolve ambiguous terms like 'it' or 'this' using session context
    context = session_memory[session_id]["context"]
    if context:
        query = re.sub(r'\bit\b|\bthis\b', context, query)
        logger.info(f"Resolved pronouns in query to: {query}")

    return query

class SQLiteConnectionPool:
    """Connection Pool to manage SQLite connections efficiently."""
    def __init__(self, db_path: str, pool_size: int = 5):
        self.db_path = db_path
        self.pool = Queue(maxsize=pool_size)
        self.lock = Lock()

        # Pre-populate the pool
        for _ in range(pool_size):
            conn = self._create_connection()
            self.pool.put(conn)

    def _create_connection(self) -> Connection:
        """Create a new SQLite connection."""
        try:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            logger.info("New SQLite connection created.")
            return conn
        except sqlite3.Error as e:
            logger.error(f"Failed to create a SQLite connection: {e}")
            raise

    def get_connection(self) -> Connection:
        """Get a connection from the pool."""
        with self.lock:
            if not self.pool.empty():
                conn = self.pool.get()
                # Validate the connection before returning
                try:
                    conn.execute("SELECT 1")
                    return conn
                except sqlite3.Error:
                    logger.warning("Connection invalid; creating a new one.")
                    return self._create_connection()
            else:
                # Create a new connection if the pool is empty
                return self._create_connection()

    def release_connection(self, conn: Connection):
        """Release a connection back to the pool."""
        with self.lock:
            try:
                # Validate connection before putting it back in the pool
                conn.execute("SELECT 1")
                if not self.pool.full():
                    self.pool.put(conn)
                else:
                    conn.close()  # Close the connection if the pool is full
                    logger.info("Connection closed as the pool is full.")
            except sqlite3.Error:
                logger.warning("Invalid connection; discarding instead of releasing.")
                conn.close()

    def close_all_connections(self):
        """Close all connections in the pool."""
        with self.lock:
            while not self.pool.empty():
                conn = self.pool.get()
                conn.close()
            logger.info("All SQLite connections closed.")


# Initialize the SQLite connection pool
DB_POOL = SQLiteConnectionPool(DB_PATH)


# Utility functions
def connect_db() -> Connection:
    """Fetch a connection from the pool."""
    try:
        conn = DB_POOL.get_connection()
        logger.debug("Database connection acquired.")
        return conn
    except Exception as e:
        logger.error(f"Failed to get database connection: {e}")
        raise HTTPException(status_code=500, detail="Database connection failed.")

def release_db_connection(conn: Connection):
    """Release the connection back to the pool."""
    try:
        DB_POOL.release_connection(conn)
        logger.debug("Database connection released.")
    except Exception as e:
        logger.warning(f"Failed to release database connection: {e}")



async def compute_embedding(text: str):
    """Compute embedding for a given text using Sentence-BERT."""
    try:
        # Check cache for the embedding
        if text in embedding_cache:
            logger.info(f"Using cached embedding for text: {text}")
            return embedding_cache[text]

        # Compute new embedding if not cached
        model = load_sentence_bert()
        embedding = await asyncio.to_thread(model.encode, text)  # Offload to a separate thread for blocking calls
        embedding = embedding.reshape(1, -1)
        embedding_cache[text] = embedding  # Cache the computed embedding
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


def is_query_relevant(query: str, threshold: float = 0.7) -> bool:
    """Determine if a query is relevant to EPR using semantic similarity."""
    global reference_embeddings  # Use precomputed embeddings
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

def learn_keywords_from_query(question: str, session_id: str):
    """Learn keywords for long-term storage and update session context for conversational flow."""
    global DYNAMIC_KEYWORDS, keyword_frequency

    # Extract entities/topics for session context
    context_entities = extract_context_entities(question)

    # Update session context with the most recent entity
    if context_entities:
        session_memory[session_id]["context"] = context_entities[-1]  # Store last entity as the context
        logger.info(f"Session {session_id} updated with context: {session_memory[session_id]['context']}")

    # Process question using spaCy to extract keywords (for learning)
    doc = nlp(question.lower())
    keywords = set()
    for token in doc:
        if token.pos_ in {"NOUN", "PROPN"} and token.text not in STOP_WORDS:
            keywords.add(token.text)
    for ent in doc.ents:
        keywords.add(ent.text)

    # Update keyword learning storage
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


async def query_validated_qa(user_embedding, question: str):
    """Query the ValidatedQA table for the best match."""
    conn = None
    try:
        conn = connect_db()
        cursor = conn.cursor()
        start_time = time.time()

        # Fetch only required columns to minimize data transfer
        rows = await asyncio.to_thread(
            lambda: cursor.execute("SELECT question, answer, question_embedding FROM ValidatedQA").fetchall()
        )

        max_similarity = 0.0
        best_match = None

        for row in rows:
            question_vector = np.frombuffer(row[2], dtype=np.float32).reshape(1, -1)
            similarity = cosine_similarity(user_embedding, question_vector)[0][0]
            if similarity > max_similarity:
                max_similarity = similarity
                best_match = row[1]  # Answer

        query_duration = time.time() - start_time
        logger.info(f"Database query completed in {query_duration:.2f} seconds")

        #release_db_connection(conn)  # Return the connection to the pool

        if best_match:
            logger.debug(f"Database match found with similarity {max_similarity:.2f}")
        else:
            logger.debug("No database match found for the query.")

        return best_match, max_similarity, "database"
    except sqlite3.Error as e:
        logger.error(f"Database query error: {e}")
        return None, 0.0, None
    except Exception as ex:
        logger.exception(f"Unexpected error in query_validated_qa: {ex}")
        return None, 0.0, None
    finally:
        # Ensure connection release in case of unexpected errors
        try:
            release_db_connection(conn)
        except UnboundLocalError:
            pass  # Connection was not established



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

def adaptive_fuzzy_match(question: str, threshold=80) -> str:
    """
    Adaptive fuzzy match with context-based adjustments.
    Dynamically adjusts thresholds based on query complexity and context.
    """
    try:
        query_length = len(question.split())
        contains_keywords = any(keyword in question.lower() for keyword in DYNAMIC_KEYWORDS)
        adjusted_threshold = threshold - 10 if contains_keywords else threshold

        # Attempt fuzzy matching against the fallback KB
        result = process.extractOne(question, FALLBACK_KB.keys(), scorer=fuzz.ratio)
        if result:
            match, score = result[0], result[1]
            logger.info(f"Fuzzy match attempt: '{match}' with score {score}")

            # Use dynamic threshold for acceptance
            if score >= adjusted_threshold:
                logger.info(f"Fuzzy match accepted: {match} with score {score}")
                return FALLBACK_KB[match]
            else:
                logger.warning(f"Low confidence fuzzy match: {match} with score {score}")
        else:
            logger.info("No suitable fuzzy match found.")
        return None
    except Exception as e:
        logger.exception("Error during adaptive fuzzy matching.")
        return None


async def enhanced_fallback_response(question: str, session_id: str) -> str:
    """
    Enhanced fallback response with adaptive fuzzy matching and session context handling.
    """
    try:
        # Step 1: Attempt adaptive fuzzy matching
        response = adaptive_fuzzy_match(question)
        if response:
            return response

        # Step 2: Use session memory for context-aware matching
        if session_id in session_memory and session_memory[session_id]["history"]:
            context_match = process.extractOne(
                question,
                [interaction["query"] for interaction in session_memory[session_id]["history"]],
                scorer=fuzz.ratio
            )
            if context_match and context_match[1] >= 70:  # Adjust threshold for context
                logger.info(f"Context match found: {context_match[0]} with score {context_match[1]}")
                return f"I'm not sure, but here's something related: {context_match[0]}"

        # Step 3: Provide a generic fallback response
        logger.info("Using generic fallback response.")
        return "I'm sorry, I couldn't find relevant information. Could you rephrase or ask something else?"
    except Exception as e:
        logger.exception("Error during enhanced fallback response generation.")
        return "I encountered an issue while finding the best response. Please try again."

def refine_with_llama(question: str, db_answer: str) -> str:
    """
    Refine the database answer using the LLaMA model to provide a concise and direct response.
    Explicitly define EPR as Extended Producer Responsibility for better contextual understanding.
    """
    try:
        # Ensure the model and tokenizer are loaded
        if llama_model is None or llama_tokenizer is None:
            logger.error("LLaMA model or tokenizer not loaded.")
            raise RuntimeError("LLaMA model or tokenizer is not initialized.")

        # Explicitly define EPR in the prompt
        epr_definition = (
            "EPR stands for Extended Producer Responsibility, which is a policy approach where producers are responsible "
            "for the treatment or disposal of post-consumer products. It focuses on plastic waste management and compliance rules."
        )

        # Generate a concise rephrased response
        prompt = (
            f"{epr_definition}\n\n"
            f"Question: {question}\n"
            f"Answer: {db_answer}\n\n"
            "Rephrased Response:"
        )

        # Tokenize inputs and align them with the model's device
        inputs = llama_tokenizer(prompt, return_tensors="pt")
        inputs = {key: val.to(llama_model.device) for key, val in inputs.items()}

        # Generate the refined response
        outputs = llama_model.generate(
            **inputs,
            max_new_tokens=130,  # Limit the response length
            do_sample=True,
            top_k=50,
            temperature=0.7
        )

        # Decode and clean the response
        refined_response = llama_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        # Extract only the rephrased response
        if "Rephrased Response:" in refined_response:
            refined_response = refined_response.split("Rephrased Response:")[-1].strip()

        logger.info("LLaMA refinement successful.")
        return refined_response

    except RuntimeError as e:
        logger.error(f"Runtime error during LLaMA refinement: {e}")
        return db_answer  # Fallback to original database answer
    except Exception as e:
        logger.exception("Error refining response with LLaMA")
        return db_answer  # Fallback to original database answer


def build_memory_context(session_id):
    """Build context from session memory."""
    if session_id not in session_memory or not session_memory[session_id]:
        return ""
    memory_context = " ".join(
        [f"User: {interaction['query']} Bot: {interaction['response']}" for interaction in session_memory[session_id]]
    )
    logger.debug(f"Memory context for session {session_id}: {memory_context}")
    return memory_context

def extract_context_entities(text):
    """Extract key entities or topics from the given text."""
    doc = nlp(text)
    entities = [ent.text for ent in doc.ents if ent.label_ in {"ORG", "PRODUCT", "EVENT", "WORK_OF_ART"}]
    # Also include nouns as potential context entities
    nouns = [token.text for token in doc if token.pos_ in {"NOUN", "PROPN"} and token.text.lower() not in STOP_WORDS]
    return entities + nouns

def update_session_context(session_id, query, response):
    """Update session memory with new query, response, and context."""
    # Add to conversation history
    session_memory[session_id]["history"].append({
        "query": query,
        "response": response,
        "timestamp": datetime.now()
    })

    # Keep only the last 5 interactions
    if len(session_memory[session_id]["history"]) > 5:
        session_memory[session_id]["history"].pop(0)

    # Extract context entities from the response (bot answer)
    context_entities = extract_context_entities(response)
    if context_entities:
        session_memory[session_id]["context"] = context_entities[0]  # Use the first entity as context
        logger.info(f"Updated session context for {session_id}: {session_memory[session_id]['context']}")


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

def test_db_connection():
    """Test the database connection and ensure tables exist."""
    try:
        conn = connect_db()
        cursor = conn.cursor()

        # Check the existence of tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        logger.info(f"Tables in the database: {tables}")

        # Check `ValidatedQA` table structure
        if ("ValidatedQA",) in tables:
            cursor.execute("PRAGMA table_info(ValidatedQA);")
            columns = cursor.fetchall()
            logger.info(f"Columns in ValidatedQA table: {columns}")
        else:
            logger.warning("ValidatedQA table not found in the database.")

        release_db_connection(conn)
    except sqlite3.Error as e:
        logger.error(f"Database connection test failed: {e}")
    except Exception as ex:
        logger.exception(f"Unexpected error during database connection test: {ex}")

@app.get("/metrics")
async def metrics():
    """Expose Prometheus metrics via FastAPI."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/session/{session_id}")
async def get_session_details(session_id: str):
    """Endpoint to retrieve session details for debugging or review."""
    if session_id not in session_memory:
        raise HTTPException(status_code=404, detail="Session not found.")
    return session_memory[session_id]


def cache_lookup(query_embedding):
    """
    Look up the cache for a similar question and its answer from Redis or in-memory cache.
    Returns the best match and its similarity score.
    """
    try:
        # Use Redis for cache lookup
        cached_items = redis_client.hgetall("query_cache")
        max_similarity = 0.0
        best_answer = None

        for cached_question, cached_data in cached_items.items():
            cached_data = json.loads(cached_data)  # Deserialize Redis JSON
            cached_embedding = np.array(cached_data["embedding"])
            cached_answer = cached_data["answer"]

            # Calculate similarity
            similarity = cosine_similarity(query_embedding, cached_embedding.reshape(1, -1))[0][0]
            if similarity > max_similarity:
                max_similarity = similarity
                best_answer = cached_answer

        if best_answer:
            logger.info(f"Cache hit with similarity {max_similarity:.2f}")
        else:
            logger.info("Cache miss for the query.")
        return best_answer, max_similarity

    except Exception as e:
        logger.exception("Error during cache lookup")
        return None, 0.0

# Configuration for maximum session management
MAX_SESSIONS = 1000  # Maximum number of active sessions allowed in memory

def evict_oldest_sessions():
    """Evict the oldest sessions if the session count exceeds MAX_SESSIONS."""
    try:
        if len(session_memory) > MAX_SESSIONS:
            # Sort sessions by the timestamp of the last interaction
            sorted_sessions = sorted(
                session_memory.items(),
                key=lambda item: item[1][-1]["timestamp"] if item[1] else datetime.min,
                reverse=False  # Oldest first
            )
            # Calculate how many sessions to remove
            sessions_to_remove = len(session_memory) - MAX_SESSIONS

            # Remove oldest sessions
            for session_id, _ in sorted_sessions[:sessions_to_remove]:
                del session_memory[session_id]
            logger.info(f"Evicted {sessions_to_remove} oldest sessions to manage memory.")
    except Exception as e:
        logger.exception("Error during session eviction.")
    # Save updated user details to Redis
    redis_client.hset(session_key, "user_details", json.dumps(user_details))

    return user_details, next_question

# Updated /chat endpoint
@app.post("/chat")
async def chat_endpoint(request: Request):
    start_time = time.time()  # Measure response time
    try:
        # Parse request data
        data = await request.json()
        session_id = data.get("session_id", None)

        # Generate a new session ID if not provided
        if not session_id:
            session_id = str(uuid.uuid4())
            logger.info(f"Generated new session ID: {session_id}")

        # Key for session in Redis
        session_key = f"session:{session_id}"
        session_data = await asyncio.to_thread(redis_client.hgetall, session_key)

        # Initialize session if it doesn't exist
        if not session_data:
            session_data = {
                "history": json.dumps([]),  # Store history as JSON string
                "context": "",
                "last_interaction": datetime.utcnow().isoformat(),
                "user_data_collected": "false",  # Flag for user data collection
                "user_name": "",  # Placeholder for user name
            }
            await asyncio.to_thread(redis_client.hmset, session_key, session_data)
            logger.info(f"New session initialized: {session_id}")
            return JSONResponse(
                content={
                    "message": "Before we start, please provide your details.",
                    "redirect_to": "/collect_user_data",
                    "session_id": session_id,
                },
                status_code=200,
            )

        # Check if user data is collected
        if session_data.get("user_data_collected", "false") == "false":
            logger.info(f"Session {session_id}: User data not collected. Prompting for user data collection.")
            return JSONResponse(
                content={
                    "message": "Before we start, please provide your details.",
                    "redirect_to": "/collect_user_data",
                    "session_id": session_id,
                },
                status_code=200,
            )

        # Fetch user's name for personalization
        user_name = session_data.get("user_name", "").strip()

        # Check for empty message or welcome scenario
        raw_question = data.get("message", "").strip()
        if not raw_question:
            welcome_message = (
                f"Welcome back {user_name}, how can I help you today?"
                if user_name
                else "Welcome back! Please type your question."
            )
            logger.info(f"Session {session_id}: Empty message received. Sending welcome message.")
            return JSONResponse(
                content={"message": welcome_message, "session_id": session_id},
                status_code=200,
            )

        logger.info(f"Session {session_id}: Received question: {raw_question}")

        # Update session history
        history = json.loads(session_data.get("history", "[]"))
        history.append(
            {
                "query": raw_question,
                "response": "Processing...",
                "timestamp": datetime.utcnow().isoformat(),
            }
        )
        session_data["history"] = json.dumps(history)
        session_data["last_interaction"] = datetime.utcnow().isoformat()

        # Save session data to Redis
        await asyncio.to_thread(redis_client.hmset, session_key, session_data)
        await asyncio.to_thread(redis_client.expire, session_key, int(SESSION_TIMEOUT.total_seconds()))

        # Step 1: Preprocess the query
        question = preprocess_query(raw_question, session_id)

        # Step 2: Compute query embedding
        user_embedding = await compute_embedding(question)

        # Step 3: Cache lookup using Redis
        cached_answer, cached_similarity = cache_lookup(user_embedding)
        if cached_answer and cached_similarity >= CACHE_THRESHOLD:
            logger.info(f"Cache hit for session {session_id}. Similarity: {cached_similarity:.2f}")
            # Update session context with cached answer
            update_session_context(session_id, raw_question, cached_answer)
            return JSONResponse(
                content={
                    "answer": cached_answer,
                    "confidence": float(cached_similarity),
                    "source": "cache",
                    "response_time": f"{time.time() - start_time:.2f} seconds",
                }
            )

        # Step 4: Database search for the best match
        db_answer, confidence, source = await query_validated_qa(user_embedding, question)
        if db_answer and confidence >= 0.5:
            logger.info(f"Database match found for session {session_id}. Confidence: {confidence:.2f}")

            # Refine the response with LLaMA
            try:
                refined_answer = await refine_with_llama(question, db_answer)
                if not refined_answer:
                    logger.warning(f"LLaMA returned an empty response for session {session_id}. Using database answer.")
                    refined_answer = db_answer
            except Exception as llama_error:
                logger.error(f"LLaMA refinement failed for session {session_id}: {llama_error}")
                refined_answer = db_answer

            # Cache the refined response in Redis
            await asyncio.to_thread(
                redis_client.hset,
                "query_cache",
                question,
                json.dumps({"embedding": user_embedding.tolist(), "answer": refined_answer}),
            )

            # Update session context with the refined answer
            update_session_context(session_id, raw_question, refined_answer)

            return JSONResponse(
                content={
                    "answer": refined_answer,
                    "confidence": float(confidence),
                    "source": source,
                    "response_time": f"{time.time() - start_time:.2f} seconds",
                }
            )

        # Step 5: Enhanced fallback response
        fallback_response = await enhanced_fallback_response(question, session_id)
        logger.info(f"Fallback response used for session {session_id}: {fallback_response}")

        # Cache the fallback response in Redis
        await asyncio.to_thread(
            redis_client.hset,
            "query_cache",
            question,
            json.dumps({"embedding": user_embedding.tolist(), "answer": fallback_response}),
        )

        # Update session context with fallback response
        update_session_context(session_id, raw_question, fallback_response)

        return JSONResponse(
            content={
                "answer": fallback_response,
                "confidence": 0.5,
                "source": "fuzzy fallback",
                "response_time": f"{time.time() - start_time:.2f} seconds",
            }
        )

    except HTTPException as e:
        logger.warning(f"HTTP error occurred: {e.detail}")
        ERROR_COUNT.inc()  # Increment error counter for Prometheus
        raise e
    except Exception as e:
        logger.exception("Unhandled exception in /chat endpoint.")
        ERROR_COUNT.inc()  # Increment error counter for Prometheus
        raise HTTPException(status_code=500, detail="An internal server error occurred.")
    finally:
        REQUEST_LATENCY.observe(time.time() - start_time)  # Record latency explicitly


SMTP_SERVER = "smtp.gmail.com"  # Replace with your SMTP server
SMTP_PORT = 587  # Standard port for TLS
SMTP_USERNAME = "urban.ease4all@gmail.com"  # Replace with your email
SMTP_PASSWORD = "evzt sfbi itnx xrxa"  # Replace with your email password

async def send_user_data_email(user_data: dict):
    """Send user data to the specified email address."""
    try:
        # Create the email content
        subject = "User Data Collected"
        recipient = "vishal.singh@recircle.in"
        sender = SMTP_USERNAME
        body = f"""
        User Data Collected:
        Name: {user_data['name']}
        Email: {user_data['email']}
        Phone: {user_data['phone']}
        Organization: {user_data['organization']}
        """
        
        # Construct the email
        msg = MIMEMultipart()
        msg['From'] = sender
        msg['To'] = recipient
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        # Send the email
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()  # Secure the connection
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.sendmail(sender, recipient, msg.as_string())
        
        logger.info("User data email sent successfully.")
    except Exception as e:
        logger.error(f"Failed to send user data email: {e}")
        raise


@app.post("/collect_user_data")
async def collect_user_data(request: Request):
    try:
        # Parse form data
        data = await request.json()
        session_id = data.get("session_id")
        name = data.get("name")
        email = data.get("email")
        phone = data.get("phone")
        organization = data.get("organization")

        if not session_id:
            logger.error("Missing session ID in user data submission.")
            return JSONResponse(content={"message": "Session ID is required."}, status_code=400)

        # Validate required fields
        if not all([name, email, phone, organization]):
            logger.warning("Incomplete user data received.")
            return JSONResponse(
                content={"message": "Please provide all required fields: name, email, phone, and organization."},
                status_code=400,
            )

        # Perform optional validation (e.g., email and phone format)
        if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
            logger.warning("Invalid email format.")
            return JSONResponse(content={"message": "Invalid email format."}, status_code=400)
        if not re.match(r"^\+?\d{7,15}$", phone):
            logger.warning("Invalid phone number format.")
            return JSONResponse(content={"message": "Invalid phone number format."}, status_code=400)

        # Save user data to Redis
        session_key = f"session:{session_id}"
        session_data = await asyncio.to_thread(redis_client.hgetall, session_key)

        user_data = {
            "name": name,
            "email": email,
            "phone": phone,
            "organization": organization,
        }
        session_data["user_data"] = json.dumps(user_data)
        session_data["user_data_collected"] = "true"
        await asyncio.to_thread(redis_client.hmset, session_key, session_data)

        logger.info(f"User data collected for session {session_id}: {session_data['user_data']}")

        # Send the user data via email
        await send_user_data_email(user_data)

        # Acknowledge data collection
        return JSONResponse(content={"message": "User data collected successfully. You can now ask your question."})

    except Exception as e:
        logger.exception("Error in collect_user_data endpoint.")
        return JSONResponse(content={"message": "An error occurred while collecting user data."}, status_code=500)
        return JSONResponse(content={"message": "An error occurred while collecting user data."}, status_code=500)
