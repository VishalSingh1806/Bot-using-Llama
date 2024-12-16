from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from accelerate import infer_auto_device_map, init_empty_weights
import sqlite3
import numpy as np
import faiss
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
# In-memory storage for session-based memory
session_memory = defaultdict(list)  # {session_id: [(query, response), ...]}
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
    """
    Handle application startup and shutdown. Initialize resources like FAISS index,
    models, dynamic keywords, and database connections during startup.
    """
    global faiss_index, id_to_question_answer, llama_model, llama_tokenizer, reference_embeddings

    logger.info("Application startup: Initializing resources...")

    try:
        # Step 1: Test database connection
        logger.info("Testing database connection...")
        test_db_connection()

        # Step 2: Load Sentence-BERT model
        logger.info("Loading Sentence-BERT model...")
        load_sentence_bert()

        # Step 3: Precompute static reference embeddings for context queries
        try:
            logger.info("Precomputing reference embeddings...")
            reference_queries = [
                "What is EPR?",
                "Explain plastic waste management.",
                "What are EPR compliance rules?",
                "How do I register for EPR compliance?"
            ]
            reference_embeddings = np.vstack([compute_embedding(q) for q in reference_queries])
            logger.info("Static embeddings precomputed successfully.")
        except Exception as e:
            logger.error(f"Failed to precompute reference embeddings: {e}")

        # Step 4: Initialize FAISS index
        logger.info("Initializing FAISS index...")
        initialize_faiss_index()

        # Step 5: Load LLaMA model and tokenizer
        try:
            logger.info("Loading LLaMA model and tokenizer...")
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
            logger.info("LLaMA model and tokenizer loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load LLaMA model and tokenizer: {e}")
            raise RuntimeError("Model loading failed during startup.")

        # Step 6: Load dynamic keywords from file
        logger.info("Loading dynamic keywords from file...")
        load_keywords_from_file()

    except Exception as e:
        logger.exception("Error during application startup. Shutting down...")
        raise RuntimeError("Application startup failed due to resource initialization errors.")

    # On shutdown, perform cleanup tasks
    try:
        yield
    finally:
        # Save dynamic keywords to file
        try:
            logger.info("Saving dynamic keywords to file...")
            save_keywords_to_file()
        except Exception as e:
            logger.error(f"Error saving dynamic keywords during shutdown: {e}")

        # Additional cleanup can be added here if needed
        logger.info("Application shutdown: Resources cleaned up.")


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

# Initialize FAISS index
def initialize_faiss_index():
    """
    Initialize FAISS index with embeddings from the database.
    """
    global faiss_index, id_to_question_answer

    embedding_dim = 384  # Adjust this based on your Sentence-BERT model
    faiss_index = faiss.IndexFlatL2(embedding_dim)
    id_to_question_answer = {}

    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("SELECT id, question, answer, question_embedding FROM ValidatedQA")

    all_embeddings = []
    for row in cursor.fetchall():
        question_id, question, answer, embedding_blob = row
        embedding_vector = np.frombuffer(embedding_blob, dtype=np.float32)
        all_embeddings.append(embedding_vector)
        id_to_question_answer[question_id] = {"question": question, "answer": answer}

    if all_embeddings:
        faiss_index.add(np.array(all_embeddings, dtype=np.float32))

    conn.close()
    logger.info(f"FAISS index initialized with {len(all_embeddings)} embeddings.")


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

def learn_keywords_from_query(question: str):
    """Learn keywords from the user query using NLP."""
    global DYNAMIC_KEYWORDS, keyword_frequency

    # Process the question using spaCy
    doc = nlp(question.lower())

    # Extract nouns, proper nouns, and named entities as keywords
    keywords = set()
    for token in doc:
        if token.pos_ in {"NOUN", "PROPN"} and token.text not in STOP_WORDS:
            keywords.add(token.text)
    for ent in doc.ents:
        keywords.add(ent.text)

    # Update the global dynamic keyword storage and frequency
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


def query_validated_qa(user_embedding, question: str, top_k: int = 5):
    """Query the FAISS index for the closest match."""
    try:
        if faiss_index is None:
            logger.error("FAISS index is not initialized.")
            raise RuntimeError("FAISS index is not ready for queries.")

        # Perform FAISS search
        distances, indices = faiss_index.search(user_embedding, top_k)

        # Retrieve the best match
        best_match_id = indices[0][0]
        if best_match_id == -1:  # No match found
            logger.info(f"No FAISS match found for query: {question}")
            return None, 0.0, None

        # Retrieve question-answer pair using the ID
        best_match = id_to_question_answer.get(best_match_id, None)
        max_similarity = 1 - distances[0][0]  # Convert L2 distance to cosine similarity

        if best_match:
            logger.info(f"FAISS match found with similarity: {max_similarity}")
            return best_match["answer"], max_similarity, "database"

        logger.info(f"No valid match found for query: {question}")
        return None, 0.0, None

    except Exception as e:
        logger.exception("Error querying FAISS index.")
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
            max_new_tokens=100,  # Limit the response length
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
        question = preprocess_query(data.get("message", "").strip())
        session_id = data.get("session_id", "default")

        if not question:
            logger.warning("Received an empty message.")
            raise HTTPException(status_code=400, detail="Message cannot be empty.")

        # Manage sessions
        if session_id not in session_memory:
            if len(session_memory) >= MAX_SESSIONS:
                evict_oldest_sessions()  # Evict sessions if limit is exceeded
            logger.info(f"New session initialized: {session_id}")

        # Step 1: Add the query to session memory with a placeholder response
        session_memory[session_id].append({
            "query": question,
            "response": "Processing...",
            "timestamp": datetime.now()
        })

        # Limit memory to the last 5 interactions for this session
        if len(session_memory[session_id]) > 5:
            session_memory[session_id].pop(0)

        # Load dynamic keywords (no logging here to avoid repetitive logs)
        load_keywords_from_file()

        # Step 2: Compute query embedding
        user_embedding = compute_embedding(question)

        # Step 3: Cache Lookup
        cached_answer, cached_similarity = cache_lookup(user_embedding)
        if cached_answer and cached_similarity >= CACHE_THRESHOLD:
            logger.info(f"Cache hit for session {session_id}. Similarity: {cached_similarity:.2f}")
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

            # Update session memory with the final response
            session_memory[session_id][-1]["response"] = refined_answer
            learn_keywords_from_query(question)
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

        # Update session memory with fallback response
        session_memory[session_id][-1]["response"] = fallback_response
        learn_keywords_from_query(question)

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
