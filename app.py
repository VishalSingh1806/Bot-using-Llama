import os
import re
import json
import time
import logging
import psycopg2
from psycopg2 import pool
from google.cloud import secretmanager
import phonenumbers
from email_validator import validate_email, EmailNotValidError
from typing import List
from concurrent.futures import ThreadPoolExecutor
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
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



SESSION_TIMEOUT = timedelta(minutes=5)

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
CACHE_THRESHOLD = 0.9 # Minimum similarity for cache retrieval

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

def get_secret(secret_name):
    """Retrieve secret value from GCP Secret Manager."""
    client = secretmanager.SecretManagerServiceClient()
    project_id = os.getenv("GCP_PROJECT_ID")  # Dynamically retrieve project ID
    if not project_id:
        raise ValueError("GCP_PROJECT_ID environment variable is not set.")
    
    # Construct the full secret path
    name = f"projects/{project_id}/secrets/{secret_name}/versions/latest"
    
    # Retrieve the secret value
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode("UTF-8")


def load_secrets(required_secrets):
    """Fetch and validate all required secrets."""
    secrets = {}
    for secret_name in required_secrets:
        try:
            secrets[secret_name] = get_secret(secret_name)
        except Exception as e:
            logger.error(f"Failed to fetch secret: {secret_name}. Error: {e}")
            raise RuntimeError(f"Critical secret missing: {secret_name}")
    return secrets


# Define required secrets
REQUIRED_SECRETS = [
    "REDIS_HOST", "REDIS_PORT", "REDIS_DB",
    "hf_token", "smtp_username", "smtp_password",
    "db_name", "db_user", "db_password", "db_host", "db_port"
]

# Load secrets
secrets = load_secrets(REQUIRED_SECRETS)

# Assign secrets to variables
REDIS_HOST = secrets["REDIS_HOST"]
REDIS_PORT = secrets["REDIS_PORT"]
REDIS_DB = secrets["REDIS_DB"]
hf_token = secrets["hf_token"]
smtp_username = secrets["smtp_username"]
smtp_password = secrets["smtp_password"]
db_name = secrets["db_name"]
db_user = secrets["db_user"]
db_password = secrets["db_password"]
db_host = secrets["db_host"]
db_port = secrets["db_port"]

# Use the secrets in your application
logger.info(f"HF Token fetched successfully")
logger.info(f"SMTP Username fetched successfully")
logger.info(f"SMTP Password fetched successfully")
logger.info(f"HF Token fetched successfully")
logger.info(f"SMTP Username fetched successfully")
logger.info(f"SMTP Password fetched successfully")
logger.info(f"Database Host fetched successfully")
logger.info(f"REDIS_HOST fetched successfully")
logger.info(f"REDIS_PORT fetched successfully")
logger.info(f"REDIS_DB fetched successfully")

# Initialize Redis client
try:
    redis_client = redis.StrictRedis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)
    redis_client.ping()
    logger.info("Connected to Redis successfully.")
except redis.ConnectionError as e:
    logger.error("Failed to connect to Redis.")
    raise RuntimeError("Redis connection failed.") from e



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
    # Precompute and store embeddings in Redis
    await precompute_and_store_embeddings()
    # Precompute static embeddings for reference queries
    try:
        reference_queries = [
            "What is EPR?",
            "Explain plastic waste management.",
            "What are EPR compliance rules?",
            "How do I register for EPR compliance?"
        ]
        reference_embeddings = np.vstack(await asyncio.gather(*(compute_embedding(q) for q in reference_queries)))
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

# Mount static files
app.mount("/static", StaticFiles(directory=STATIC_FILES_DIR), name="static")

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

class PostgreSQLConnectionPool:
    """Connection Pool to manage PostgreSQL connections."""
    def __init__(self, db_config: dict, pool_size: int = 10):
        self.db_config = db_config
        self.pool = pool.SimpleConnectionPool(
            minconn=1,
            maxconn=pool_size,
            dbname=db_config["dbname"],
            user=db_config["user"],
            password=db_config["password"],
            host=db_config["host"],
            port=db_config["port"],
            connect_timeout=10
        )
        logger.info("PostgreSQL connection pool initialized.")

    def get_connection(self):
        """Fetch a connection from the pool."""
        try:
            conn = self.pool.getconn()
            logger.debug("PostgreSQL connection acquired.")
            return conn
        except Exception as e:
            logger.error(f"Failed to acquire PostgreSQL connection: {e}")
            raise

    def release_connection(self, conn):
        """Release the connection back to the pool."""
        try:
            self.pool.putconn(conn)
            logger.debug("PostgreSQL connection released back to pool.")
        except Exception as e:
            logger.warning(f"Failed to release PostgreSQL connection: {e}")

    def close_all_connections(self):
        """Close all connections in the pool."""
        try:
            self.pool.closeall()
            logger.info("All PostgreSQL connections closed.")
        except Exception as e:
            logger.error(f"Error closing PostgreSQL connections: {e}")



POSTGRES_CONFIG = {
    "dbname": db_name,
    "user": db_user,
    "password": db_password,
    "host": db_host,
    "port": db_port,
}

# Initialize the PostgreSQL connection pool
DB_POOL = PostgreSQLConnectionPool(POSTGRES_CONFIG)


def connect_db():
    """Fetch a connection from the PostgreSQL pool."""
    try:
        conn = DB_POOL.get_connection()
        logger.debug("Database connection acquired.")
        return conn
    except Exception as e:
        logger.error(f"Failed to get database connection: {e}")
        raise HTTPException(status_code=500, detail="Database connection failed.")


def release_db_connection(conn):
    """Release the connection back to the PostgreSQL pool."""
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
    """Query the validatedqa table for the best match."""
    conn = None
    try:
        conn = connect_db()
        if conn is None:
            logger.error("Database connection is None. Cannot proceed with query.")
            return None, 0.0, None

        cursor = conn.cursor()
        if cursor is None:
            logger.error("Failed to create a cursor from the database connection.")
            return None, 0.0, None

        start_time = time.time()

        # Fetch only required columns to minimize data transfer
        rows = await asyncio.to_thread(lambda: fetch_rows_postgresql(cursor))

        if not rows:
            logger.info("No rows returned from the database query.")
            return None, 0.0, None

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

        if best_match:
            logger.debug(f"Database match found with similarity {max_similarity:.2f}")
        else:
            logger.debug("No database match found for the query.")

        return best_match, max_similarity, "database"
    except psycopg2.Error as e:
        logger.error(f"PostgreSQL query error: {e}")
        return None, 0.0, None
    except Exception as ex:
        logger.exception(f"Unexpected error in query_validated_qa: {ex}")
        return None, 0.0, None
    finally:
        # Ensure connection release in case of unexpected errors
        if conn:
            try:
                release_db_connection(conn)
            except Exception as release_error:
                logger.warning(f"Error releasing database connection: {release_error}")


def fetch_rows_postgresql(cursor):
    """Fetch rows from the validatedqa table."""
    try:
        cursor.execute("SELECT question, answer, question_embedding FROM validatedqa")
        rows = cursor.fetchall()
        return rows
    except Exception as e:
        logger.error(f"Error fetching rows: {e}")
        return None


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
    """Test the PostgreSQL database connection and ensure tables exist."""
    try:
        logger.info("Starting PostgreSQL database connection test...")

        # Attempt to connect to the database
        conn = connect_db()
        if not conn:
            logger.error("Failed to establish a database connection.")
            return

        logger.info("Database connection established successfully.")
        cursor = conn.cursor()

        # Check the existence of tables in the database
        logger.info("Checking for tables in the database...")
        cursor.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public';
        """)
        tables = cursor.fetchall()
        if tables:
            logger.info(f"Tables found in the database: {[table[0] for table in tables]}")
        else:
            logger.warning("No tables found in the database.")

        # Check if the `validatedqa` table exists
        if ("validatedqa",) in tables:
            logger.info("The 'validatedqa' table exists. Checking its structure...")

            # Retrieve and log the columns in the `validatedqa` table
            cursor.execute("""
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_name = 'validatedqa';
            """)
            columns = cursor.fetchall()
            if columns:
                logger.info(f"Columns in 'validatedqa' table: {columns}")
            else:
                logger.warning("The 'validatedqa' table has no columns or is inaccessible.")

            # Check if the `validatedqa` table contains any data
            logger.info("Checking for data in the 'validatedqa' table...")
            cursor.execute("SELECT COUNT(*) FROM validatedqa;")
            row_count = cursor.fetchone()[0]
            if row_count > 0:
                logger.info(f"The 'validatedqa' table contains {row_count} rows.")

                # Fetch a sample row for verification
                cursor.execute("SELECT * FROM validatedqa LIMIT 1;")
                sample_row = cursor.fetchone()
                logger.info(f"Sample row from 'validatedqa': {sample_row}")
            else:
                logger.warning("The 'validatedqa' table is empty.")
        else:
            logger.warning("The 'validatedqa' table does not exist in the database.")

        # Release the database connection
        release_db_connection(conn)
        logger.info("Database connection test completed successfully.")

    except psycopg2.Error as e:
        logger.error(f"PostgreSQL connection test failed with an error: {e}")
    except Exception as ex:
        logger.exception(f"Unexpected error during PostgreSQL connection test: {ex}")
    finally:
        # Ensure the connection is released even if an error occurs
        try:
            if conn:
                release_db_connection(conn)
                logger.info("Database connection released.")
        except Exception as release_error:
            logger.warning(f"Error while releasing database connection: {release_error}")

@app.get("/session/{session_id}")
async def get_session_details(session_id: str):
    """Endpoint to retrieve session details for debugging or review."""
    if session_id not in session_memory:
        raise HTTPException(status_code=404, detail="Session not found.")
    return session_memory[session_id]


# def cache_lookup(query_embedding, threshold=0.9):
#     """
#     Look up the cache for a similar question and its answer from Redis or in-memory cache.
#     Returns the best match and its similarity score.
#     """
#     try:
#         max_similarity = 0.0
#         best_answer = None

#         # Efficiently iterate over cached items in Redis
#         for key in redis_client.scan_iter("query_cache:*"):
#             cached_data = redis_client.hgetall(key)
#             if "embedding" in cached_data and "answer" in cached_data:
#                 cached_embedding = np.array(json.loads(cached_data["embedding"]))
#                 cached_answer = cached_data["answer"]

#                 # Calculate similarity
#                 similarity = cosine_similarity(query_embedding, cached_embedding.reshape(1, -1))[0][0]
#                 if similarity > max_similarity and similarity >= threshold:
#                     max_similarity = similarity
#                     best_answer = cached_answer

#         if best_answer:
#             logger.info(f"Cache hit with similarity {max_similarity:.2f}, Answer: {best_answer}")
#         else:
#             logger.info("Cache miss for the query.")

#         return best_answer, max_similarity

#     except Exception as e:
#         logger.exception("Error during cache lookup")
#         return None, 0.0

def cache_lookup(query_embedding, threshold=0.9):
    """
    Look up the cache for a similar question and its answer from Redis.
    Returns the best match and its similarity score.
    """
    try:
        max_similarity = 0.0
        best_answer = None

        # Iterate over cached items in Redis
        for key in redis_client.scan_iter("query_cache:*"):
            cached_data = redis_client.hgetall(key)
            if "embedding" in cached_data and "answer" in cached_data:
                cached_embedding = np.array(json.loads(cached_data["embedding"]))
                cached_answer = cached_data["answer"]

                # Calculate similarity
                similarity = cosine_similarity(query_embedding, cached_embedding.reshape(1, -1))[0][0]
                if similarity > max_similarity and similarity >= threshold:
                    max_similarity = similarity
                    best_answer = cached_answer

        if best_answer:
            logger.info(f"Cache hit with similarity {max_similarity:.2f}, Answer: {best_answer}")
        else:
            logger.info("Cache miss for the query.")

        return best_answer, max_similarity

    except Exception as e:
        logger.exception("Error during cache lookup.")
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

# Updated /chat endpoint with consistent LLaMA refinement
# Updated /chat endpoint with consistent LLaMA refinement
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
            logger.info(f"Cache hit for session {session_id}. Similarity: {cached_similarity:.2f}, Answer: {cached_answer}")

            # Refine cached answer with LLaMA
            refined_answer = refine_with_llama(question, cached_answer)
            update_session_context(session_id, raw_question, refined_answer)

            return JSONResponse(
                content={
                    "answer": refined_answer,
                    "confidence": float(cached_similarity),
                    "source": "cache",
                    "response_time": f"{time.time() - start_time:.2f} seconds",
                }
            )

        # Step 4: Database search for the best match
        db_answer, confidence, source = await query_validated_qa(user_embedding, question)
        if db_answer and confidence >= 0.5:
            logger.info(f"Database match found for session {session_id}. Confidence: {confidence:.2f}, Answer: {db_answer}")

            # Refine the response with LLaMA
            refined_answer = refine_with_llama(question, db_answer)

            # Cache the refined response in Redis
            await asyncio.to_thread(
                redis_client.hset,
                f"query_cache:{question}",
                mapping={
                    "embedding": json.dumps(user_embedding.tolist()),
                    "answer": refined_answer,
                },
            )
            logger.info(f"Cached question: {question} with answer: {refined_answer}")

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
        refined_fallback = refine_with_llama(question, fallback_response)

        # Cache the fallback response in Redis
        await asyncio.to_thread(
            redis_client.hset,
            f"query_cache:{question}",
            mapping={
                "embedding": json.dumps(user_embedding.tolist()),
                "answer": refined_fallback,
            },
        )
        logger.info(f"Fallback response cached for session {session_id}: {refined_fallback}")

        update_session_context(session_id, raw_question, refined_fallback)

        return JSONResponse(
            content={
                "answer": refined_fallback,
                "confidence": 0.5,
                "source": "fuzzy fallback",
                "response_time": f"{time.time() - start_time:.2f} seconds",
            }
        )

    except Exception as e:
        logger.exception("Unhandled exception in /chat endpoint.")
        return JSONResponse(
            content={"message": "An internal server error occurred."}, status_code=500
        )
    finally:
        REQUEST_LATENCY.observe(time.time() - start_time)


# Updated /chat endpoint (this doesn't refine cached data)
# @app.post("/chat")
# async def chat_endpoint(request: Request):
#     start_time = time.time()  # Measure response time
#     try:
#         # Parse request data
#         data = await request.json()
#         session_id = data.get("session_id", None)

#         # Generate a new session ID if not provided
#         if not session_id:
#             session_id = str(uuid.uuid4())
#             logger.info(f"Generated new session ID: {session_id}")

#         # Key for session in Redis
#         session_key = f"session:{session_id}"
#         session_data = await asyncio.to_thread(redis_client.hgetall, session_key)

#         # Initialize session if it doesn't exist
#         if not session_data:
#             session_data = {
#                 "history": json.dumps([]),  # Store history as JSON string
#                 "context": "",
#                 "last_interaction": datetime.utcnow().isoformat(),
#                 "user_data_collected": "false",  # Flag for user data collection
#                 "user_name": "",  # Placeholder for user name
#             }
#             await asyncio.to_thread(redis_client.hmset, session_key, session_data)
#             logger.info(f"New session initialized: {session_id}")
#             return JSONResponse(
#                 content={
#                     "message": "Before we start, please provide your details.",
#                     "redirect_to": "/collect_user_data",
#                     "session_id": session_id,
#                 },
#                 status_code=200,
#             )

#         # Check if user data is collected
#         if session_data.get("user_data_collected", "false") == "false":
#             logger.info(f"Session {session_id}: User data not collected. Prompting for user data collection.")
#             return JSONResponse(
#                 content={
#                     "message": "Before we start, please provide your details.",
#                     "redirect_to": "/collect_user_data",
#                     "session_id": session_id,
#                 },
#                 status_code=200,
#             )

#         # Fetch user's name for personalization
#         user_name = session_data.get("user_name", "").strip()

#         # Check for empty message or welcome scenario
#         raw_question = data.get("message", "").strip()
#         if not raw_question:
#             welcome_message = (
#                 f"Welcome back {user_name}, how can I help you today?"
#                 if user_name
#                 else "Welcome back! Please type your question."
#             )
#             logger.info(f"Session {session_id}: Empty message received. Sending welcome message.")
#             return JSONResponse(
#                 content={"message": welcome_message, "session_id": session_id},
#                 status_code=200,
#             )

#         logger.info(f"Session {session_id}: Received question: {raw_question}")

#         # Update session history
#         history = json.loads(session_data.get("history", "[]"))
#         history.append(
#             {
#                 "query": raw_question,
#                 "response": "Processing...",
#                 "timestamp": datetime.utcnow().isoformat(),
#             }
#         )
#         session_data["history"] = json.dumps(history)
#         session_data["last_interaction"] = datetime.utcnow().isoformat()

#         # Save session data to Redis
#         await asyncio.to_thread(redis_client.hmset, session_key, session_data)
#         await asyncio.to_thread(redis_client.expire, session_key, int(SESSION_TIMEOUT.total_seconds()))

#         # Step 1: Preprocess the query
#         question = preprocess_query(raw_question, session_id)

#         # Step 2: Compute query embedding
#         user_embedding = await compute_embedding(question)

#         # Step 3: Cache lookup using Redis
#         cached_answer, cached_similarity = cache_lookup(user_embedding)
#         if cached_answer and cached_similarity >= CACHE_THRESHOLD and cached_answer.strip():
#             logger.info(f"Cache hit for session {session_id}. Similarity: {cached_similarity:.2f}, Answer: {cached_answer}")
#             update_session_context(session_id, raw_question, cached_answer)
#             return JSONResponse(
#                 content={
#                     "answer": cached_answer,
#                     "confidence": float(cached_similarity),
#                     "source": "cache",
#                     "response_time": f"{time.time() - start_time:.2f} seconds",
#                 }
#             )
#         else:
#             logger.warning(f"Cache hit but no valid answer or low similarity. Similarity: {cached_similarity:.2f}, Answer: {cached_answer}")

#         # Step 4: Database search for the best match
#         db_answer, confidence, source = await query_validated_qa(user_embedding, question)
#         if db_answer and confidence >= 0.5 and db_answer.strip():
#             logger.info(f"Database match found for session {session_id}. Confidence: {confidence:.2f}, Answer: {db_answer}")

#             # Refine the response with LLaMA
#             try:
#                 refined_answer = await refine_with_llama(question, db_answer)
#                 if not refined_answer:
#                     logger.warning(f"LLaMA returned an empty response for session {session_id}. Using database answer.")
#                     refined_answer = db_answer
#             except Exception as llama_error:
#                 logger.error(f"LLaMA refinement failed for session {session_id}: {llama_error}")
#                 refined_answer = db_answer

#             # Cache the refined response in Redis
#             await asyncio.to_thread(
#                 redis_client.hset,
#                 f"query_cache:{question}",
#                 mapping={
#                     "embedding": json.dumps(user_embedding.tolist()),
#                     "answer": refined_answer,
#                 },
#             )
#             logger.info(f"Cached question: {question} with answer: {refined_answer}")



#             # Update session context with the refined answer
#             update_session_context(session_id, raw_question, refined_answer)

#             return JSONResponse(
#                 content={
#                     "answer": refined_answer,
#                     "confidence": float(confidence),
#                     "source": source,
#                     "response_time": f"{time.time() - start_time:.2f} seconds",
#                 }
#             )
#         else:
#             logger.warning(f"Database search did not yield a valid answer. Confidence: {confidence:.2f}, Answer: {db_answer}")

#         # Step 5: Enhanced fallback response
#         logger.info(f"Fallback triggered for session {session_id}. No valid cache or database match found.")
#         fallback_response = await enhanced_fallback_response(question, session_id)
#         logger.info(f"Fallback response used for session {session_id}: {fallback_response}")

#         # Cache the fallback response in Redis
#         await asyncio.to_thread(
#             redis_client.hset,
#             "query_cache",
#             question,
#             json.dumps({"embedding": user_embedding.tolist(), "answer": fallback_response}),
#         )

#         # Update session context with fallback response
#         update_session_context(session_id, raw_question, fallback_response)

#         return JSONResponse(
#             content={
#                 "answer": fallback_response,
#                 "confidence": 0.5,
#                 "source": "fuzzy fallback",
#                 "response_time": f"{time.time() - start_time:.2f} seconds",
#             }
#         )

#     except HTTPException as e:
#         logger.warning(f"HTTP error occurred: {e.detail}")
#         ERROR_COUNT.inc()  # Increment error counter for Prometheus
#         raise e
#     except Exception as e:
#         logger.exception("Unhandled exception in /chat endpoint.")
#         ERROR_COUNT.inc()  # Increment error counter for Prometheus
#         raise HTTPException(status_code=500, detail="An internal server error occurred.")
#     finally:
#         REQUEST_LATENCY.observe(time.time() - start_time)  # Record latency explicitly


SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

email_batch = []  # Initialize the global email batch list
batch_size = 10    # Adjust batch size if needed

def construct_email(user_data):
    """Construct an email."""
    msg = MIMEMultipart()
    msg['From'] = smtp_username
    msg['To'] = "vishal.singh@recircle.in"  # Replace with recipient's email address
    msg['Subject'] = "User Data Collected"
    body = f"""
    User Data Collected:
    Name: {user_data['name']}
    Email: {user_data['email']}
    Phone: {user_data['phone']}
    Organization: {user_data['organization']}
    """
    msg.attach(MIMEText(body, 'plain'))
    return msg

def send_email(user_data):
    """Send a single email."""
    try:
        msg = construct_email(user_data)
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=30) as server:
            server.starttls()
            server.login(smtp_username, smtp_password)
            server.sendmail(msg['From'], msg['To'], msg.as_string())
        logger.info(f"Email sent successfully for {user_data['name']}")
    except Exception as e:
        logger.error(f"Failed to send email for {user_data['name']}: {e}")
        raise

def send_email_batch(email_batch):
    """Send a batch of emails using a single SMTP connection."""
    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=30) as server:
            server.starttls()
            server.login(smtp_username, smtp_password)
            for user_data in email_batch:
                try:
                    msg = construct_email(user_data)
                    server.sendmail(msg['From'], msg['To'], msg.as_string())
                    logger.info(f"Email sent successfully for {user_data['name']}")
                except Exception as e:
                    logger.error(f"Failed to send email for {user_data['name']}: {e}")
                    continue
    except smtplib.SMTPServerDisconnected as e:
        logger.error("SMTP server disconnected unexpectedly during batch.")
    except Exception as e:
        logger.error(f"Error during batch email sending: {e}")
        raise

async def collect_and_send_user_data(user_data):
    global email_batch

    # Add user data to the batch
    email_batch.append(user_data)

    # Log the current batch size for debugging
    logger.info(f"Current batch size: {len(email_batch)}")

    # If batch size is reached, send the emails
    if len(email_batch) >= batch_size:
        try:
            logger.info(f"Sending batch of {len(email_batch)} emails.")
            send_email_batch(email_batch)  # Ensure this function is working as expected
            email_batch.clear()  # Clear batch after sending
        except Exception as e:
            logger.error(f"Error sending email batch: {e}")
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

        # Validate required fields
        if not all([session_id, name, email, phone, organization]):
            logger.warning("Incomplete user data received.")
            return JSONResponse(
                content={"message": "Please provide all required fields."},
                status_code=400,
            )

        # Save user data to Redis
        await asyncio.to_thread(
            redis_client.hset,
            f"session:{session_id}",
            mapping={
                "user_data_collected": "true",
                "user_name": name,
                "email": email,
                "phone": phone,
                "organization": organization,
                "last_interaction": datetime.utcnow().isoformat(),
            },
        )
        logger.info(f"User data saved for session {session_id}.")

        # Add user data to the email batch
        await collect_and_send_user_data(
            {"name": name, "email": email, "phone": phone, "organization": organization}
        )

        return JSONResponse(
            content={"message": "User data collected successfully. You can now ask your question."},
            status_code=200,
        )

    except Exception as e:
        logger.exception("Error in collect_user_data endpoint.")
        return JSONResponse(
            content={"message": "An error occurred while collecting user data."},
            status_code=500,
        )

async def precompute_and_store_embeddings():
    """Fetch questions and answers from the database, compute embeddings, and store them in Redis."""
    try:
        conn = connect_db()
        cursor = conn.cursor()

        # Fetch all questions and answers from the database
        cursor.execute("SELECT question, answer FROM validatedqa")
        rows = cursor.fetchall()

        if not rows:
            logger.warning("No data found in the database to precompute embeddings.")
            return

        model = load_sentence_bert()

        for row in rows:
            question, answer = row

            # Compute embedding for the question
            embedding = model.encode(question).tolist()

            # Store question, answer, and embedding in Redis
            redis_client.hset(
                f"query_cache:{question}",
                mapping={
                    "answer": answer,
                    "embedding": json.dumps(embedding)
                }
            )
            logger.info(f"Cached question: {question}")

        release_db_connection(conn)
        logger.info("Precomputed embeddings stored in Redis successfully.")

    except Exception as e:
        logger.exception("Error during precomputing and storing embeddings in Redis.")

