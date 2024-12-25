from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from accelerate import infer_auto_device_map, init_empty_weights
import sqlite3
import psycopg2
import numpy as np
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
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

INITIAL_QUESTIONS = [
    "What is your name?",
    "What is your email?",
    "What is your phone number?",
    "What is your organization name?"
]

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

# Load spaCy model globally during startup
nlp = spacy.load("en_core_web_sm")

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
LOG_LEVEL = logging.INFO  # Set to DEBUG for detailed logs in development
logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app_log.log"),
        logging.StreamHandler()
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


# Updated connect_db function for PostgreSQL
def connect_db():
    """Connect to the GCP PostgreSQL database."""
    try:
        conn = psycopg2.connect(
            dbname="epr_database",
            user="postgres",
            password="Tech123",
            host="34.100.134.186",
            port="5432"
        )
        return conn
    except psycopg2.Error as e:
        logger.error(f"Database connection error: {e}")
        raise HTTPException(status_code=500, detail="Database connection failed.")

def compute_embedding(text: str):
    """Compute embedding for a given text using Sentence-BERT."""
    try:
        # Check cache for the embedding
        if text in embedding_cache:
            logger.info(f"Using cached embedding for text: {text}")
            return embedding_cache[text]

        # Compute new embedding if not cached
        model = load_sentence_bert()
        embedding = model.encode(text).reshape(1, -1)
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
        
def validate_email(email: str) -> bool:
    """Validate email format."""
    import re
    pattern = r"[^@]+@[^@]+\.[^@]+"
    return re.match(pattern, email) is not None

def validate_phone(phone: str) -> bool:
    """Validate phone number format."""
    return phone.isdigit() and len(phone) in [10, 11]

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


def query_validated_qa(user_embedding, question: str):
    """Query the ValidatedQA table for the best match, including related sections."""
    try:
        conn = connect_db()
        cursor = conn.cursor()
        
        # Fetch QA pairs with embeddings and section IDs
        cursor.execute("SELECT id, question, answer, question_embedding, section_id FROM ValidatedQA")
        qa_pairs = cursor.fetchall()

        max_similarity = 0.0
        best_match = None
        best_section = None

        for qa_id, db_question, db_answer, db_question_embedding, section_id in qa_pairs:
            question_vector = np.frombuffer(db_question_embedding, dtype=np.float32).reshape(1, -1)
            similarity = cosine_similarity(user_embedding, question_vector)[0][0]
            if similarity > max_similarity:
                max_similarity = similarity
                best_match = (qa_id, db_answer, section_id)

        if best_match:
            qa_id, db_answer, section_id = best_match
            # Fetch section content if section_id is available
            if section_id:
                cursor.execute("SELECT content FROM Sections WHERE id = %s", (section_id,))
                section_result = cursor.fetchone()
                best_section = section_result[0] if section_result else None

        conn.close()

        if best_match:
            # Combine answer with section content for a comprehensive response
            combined_answer = f"{db_answer}\n\nAdditional Context:\n{best_section}" if best_section else db_answer
            return combined_answer, max_similarity, "database"
        else:
            return None, 0.0, None
    except psycopg2.Error as e:
        logger.error(f"Database query error: {e}")
        return None, 0.0, None


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
        # Analyze query complexity (e.g., length, presence of keywords)
        query_length = len(question.split())
        contains_keywords = any(keyword in question.lower() for keyword in DYNAMIC_KEYWORDS)
        adjusted_threshold = threshold - 10 if contains_keywords else threshold

        # Attempt fuzzy matching against the knowledge base
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


def enhanced_fallback_response(question: str, session_id: str) -> str:
    try:
        # 1. Attempt fuzzy matching against the knowledge base
        response = adaptive_fuzzy_match(question)
        if response:
            return response

        # 2. Check recent session memory for context-based matching
        if session_id in session_memory and session_memory[session_id]:
            context_match = process.extractOne(
                question,
                [interaction["query"] for interaction in session_memory[session_id]],
                scorer=fuzz.ratio
            )
            if context_match and context_match[1] >= 70:  # Lower threshold for session context
                # Avoid self-repetition
                if context_match[0] != question:
                    return f"I'm not sure, but here's something related: {context_match[0]}"

        # 3. Provide a generic fallback response
        logger.info("Using generic fallback response.")
        return "I'm sorry, I couldn't find relevant information. Could you rephrase or ask something else?"
    except Exception as e:
        logger.exception("Error during enhanced fallback response generation.")
        return "I encountered an issue while finding the best response. Please try again."


def refine_with_llama(question: str, db_answer: str) -> str:
    """
    Refine the database answer using the LLaMA model to provide a concise and direct response.
    """
    try:
        # Explicitly define EPR in the prompt
        epr_definition = (
            "EPR stands for Extended Producer Responsibility, which is a policy approach where producers are responsible "
            "for the treatment or disposal of post-consumer products. It focuses on plastic waste management and compliance rules."
        )

        # Include section context in the refinement
        prompt = (
            f"{epr_definition}\n\n"
            f"Question: {question}\n"
            f"Answer: {db_answer}\n\n"
            "Rephrased Response:"
        )

        inputs = llama_tokenizer(prompt, return_tensors="pt")
        inputs = {key: val.to(llama_model.device) for key, val in inputs.items()}

        outputs = llama_model.generate(
            **inputs,
            max_new_tokens=130,
            do_sample=True,
            top_k=50,
            temperature=0.7
        )

        refined_response = llama_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        return refined_response.split("Rephrased Response:")[-1].strip()
    except Exception as e:
        logger.error(f"Error refining response with LLaMA: {e}")
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
    """Test database connection and structure."""
    try:
        # Attempt to connect to the database
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Check tables in the database
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print("Tables in the database:", tables)

        # Verify structure of the `ValidatedQA` table
        if ("ValidatedQA",) in tables:
            cursor.execute("PRAGMA table_info(ValidatedQA);")
            columns = cursor.fetchall()
            print("Columns in ValidatedQA table:", columns)
        else:
            print("ValidatedQA table not found in the database.")

        conn.close()
    except sqlite3.DatabaseError as e:
        print("Database connection error:", e)
    except Exception as ex:
        print("Unexpected error:", ex)

def cache_lookup(query_embedding):
    """Look up the cache for a similar question and its answer."""
    max_similarity = 0.0
    best_answer = None

    for cached_question, (cached_embedding, cached_answer) in CACHE.items():
        similarity = cosine_similarity(query_embedding, cached_embedding)[0][0]
        if similarity > max_similarity:
            max_similarity = similarity
            best_answer = cached_answer

    if best_answer:
        logger.debug(f"Cache hit with similarity {max_similarity:.2f}")
    else:
        logger.debug("Cache miss for the query.")

    return best_answer, max_similarity


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


@app.post("/chat")
async def chat_endpoint(request: Request):
    try:
        start_time = time.time()

        # Parse request data
        data = await request.json()
        raw_message = data.get("message", "").strip()
        session_id = data.get("session_id", "default")

        if not raw_message:
            logger.warning("Received an empty message.")
            raise HTTPException(status_code=400, detail="Message cannot be empty.")

        # Initialize session if new
        if session_id not in session_memory:
            if len(session_memory) >= MAX_SESSIONS:
                evict_oldest_sessions()
            session_memory[session_id] = {
                "history": [],
                "context": "",
                "pending_questions": INITIAL_QUESTIONS.copy(),
                "user_info": {}
            }
            logger.info(f"New session initialized: {session_id}")

        session = session_memory[session_id]

        # Handle pending initial questions
        if session["pending_questions"]:
            next_question = session["pending_questions"].pop(0)

            # Save the response to the previous question if applicable
            if session["history"]:
                last_question = session["history"][-1]["response"]
                if last_question in INITIAL_QUESTIONS:
                    key = last_question.lower().replace(" ", "_").replace("?", "")
                    session["user_info"][key] = raw_message

                    # Validate specific responses
                    if "email" in last_question.lower() and not validate_email(raw_message):
                        return {"answer": "Invalid email. Please provide a valid email address."}
                    if "phone" in last_question.lower() and not validate_phone(raw_message):
                        return {"answer": "Invalid phone number. Please provide a valid phone number."}

            # Add the next question to the session history and return it
            session["history"].append({
                "query": raw_message,
                "response": next_question,
                "timestamp": datetime.now()
            })

            return {
                "answer": next_question,
                "confidence": 1.0,
                "source": "predefined_questions",
                "response_time": f"{time.time() - start_time:.2f} seconds",
            }

        # All initial questions answered, summarize user info
        if not session["pending_questions"] and not session.get("summary_shown", False):
            session["summary_shown"] = True
            return {
                "answer": f"Thank you! Here's what I collected: {session['user_info']}. "
                          f"You can now ask your queries.",
                "confidence": 1.0,
                "source": "summary",
                "response_time": f"{time.time() - start_time:.2f} seconds",
            }

        # Preprocess query and resolve ambiguous references using session context
        question = preprocess_query(raw_message, session_id)

        # Step 1: Add the query to session memory with a placeholder response
        session["history"].append({
            "query": raw_message,
            "response": "Processing...",
            "timestamp": datetime.now()
        })

        # Limit history to the last 5 interactions
        if len(session["history"]) > 5:
            session["history"].pop(0)

        # Step 2: Compute query embedding
        user_embedding = compute_embedding(question)

        # Step 3: Cache Lookup
        cached_answer, cached_similarity = cache_lookup(user_embedding)
        if cached_answer and cached_similarity >= CACHE_THRESHOLD:
            logger.info(f"Cache hit for session {session_id}. Similarity: {cached_similarity:.2f}")
            update_session_context(session_id, raw_message, cached_answer)
            return {
                "answer": cached_answer,
                "confidence": float(cached_similarity),
                "source": "cache",
                "response_time": f"{time.time() - start_time:.2f} seconds",
            }

        # Step 4: Database Search for Best Match
        db_answer, confidence, source = query_validated_qa(user_embedding, question)

        if db_answer and confidence >= 0.5:  # Database confidence threshold
            logger.info(f"Database match found for session {session_id}. Confidence: {confidence:.2f}")

            # Refine the response with LLaMA
            try:
                refined_answer = refine_with_llama(question, db_answer)
                if not refined_answer:
                    logger.warning(f"LLaMA returned an empty response for session {session_id}. Using database answer.")
                    refined_answer = db_answer
            except Exception as llama_error:
                logger.error(f"LLaMA refinement failed for session {session_id}: {llama_error}")
                refined_answer = db_answer

            # Update cache with refined answer
            CACHE[question] = (user_embedding, refined_answer)

            # Update session context with query and refined answer
            update_session_context(session_id, raw_message, refined_answer)

            return {
                "answer": refined_answer,
                "confidence": float(confidence),
                "source": source,
                "response_time": f"{time.time() - start_time:.2f} seconds",
            }

        # Step 5: Enhanced Fallback Response
        fallback_response = enhanced_fallback_response(question, session_id)
        logger.info(f"Fallback response used for session {session_id}: {fallback_response}")

        # Update cache with fallback response
        CACHE[question] = (user_embedding, fallback_response)

        # Update session context with query and fallback response
        update_session_context(session_id, raw_message, fallback_response)

        return {
            "answer": fallback_response,
            "confidence": 0.5,
            "source": "fuzzy fallback",
            "response_time": f"{time.time() - start_time:.2f} seconds",
        }

    except HTTPException as e:
        logger.warning(f"HTTP error occurred: {e.detail}")
        raise e
    except TypeError as te:
        logger.error(f"Serialization error during response: {te}")
        raise HTTPException(status_code=500, detail="Response serialization error.")
    except Exception as e:
        logger.exception("Unhandled exception in /chat endpoint.")
        raise HTTPException(status_code=500, detail="An internal server error occurred.")
