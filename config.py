import os
from dotenv import load_dotenv
from datetime import date
import streamlit as st # Import streamlit

# Load environment variables
load_dotenv()

# --- Get Absolute Path Base ---
# Get the directory where this config.py file is located
_config_dir = os.path.dirname(__file__)

# --- General Config ---
# Try to get GOOGLE_API_KEY from Streamlit secrets, otherwise from .env
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except: 
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

try:
    LANGCHAIN_API_KEY = st.secrets["LANGCHAIN_API_KEY"]
except:
    LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")

try:
    LANGCHAIN_TRACING_V2 = st.secrets["LANGCHAIN_TRACING_V2"]
except:
    LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2")

LLM_MODEL_NAME = "gemini-1.5-pro-latest" # Or your preferred model
EMBEDDING_MODEL_NAME = "models/text-embedding-004"
AGENT_IS_VERBOSE = True # Set to True for detailed agent logging

# --- Google API Scopes ---
# Define all scopes needed by any tool in one place
COMBINED_SCOPES = [
    'https://www.googleapis.com/auth/gmail.readonly',      # For email_downloader
    'https://www.googleapis.com/auth/documents.readonly',   # For google_doc_tool
    'https://www.googleapis.com/auth/spreadsheets.readonly' # For google_sheet_tool
]

# --- Handbook Config ---
# Ensure PDF path is correct relative to config.py or use absolute
PDF_DOC_PATH = os.path.abspath(os.path.join(_config_dir, "CATE Handbook 2024_August 21.pdf"))
# Use absolute path for the DB directory
HANDBOOK_DB_DIRECTORY = os.path.abspath(os.path.join(_config_dir, "chroma_db_handbook_google_emb"))
REBUILD_HANDBOOK_VECTOR_STORE = False # Set to True to force rebuild on startup
HANDBOOK_CHUNK_SIZE = 700
HANDBOOK_CHUNK_OVERLAP = 70

# --- Email Config ---
# Use absolute path for the DB directory
EMAIL_DB_DIRECTORY = os.path.abspath(os.path.join(_config_dir, "chroma_db_emails_google_text_embedding_004"))
REBUILD_EMAIL_VECTOR_STORE = True # Set to True to force rebuild on startup
USE_EMAIL_SELF_QUERY = False # Set to True to use self-querying retriever for emails
UPDATE_EMAIL_ON_STARTUP = True # Set to True to check for new emails when the app starts
EMAIL_UPDATE_WINDOW_DAYS = 2   # How many days back to check for new emails (on startup or manual update)
DUMP_EMAIL_METADATA_ON_STARTUP = False # Set to True for debugging metadata on startup
# Optional: Add timeframe for initial build if rebuilding
TIME_FRAME_FOR_EMAIL_VECTOR_STORE_BUILD = '60d' # e.g., '30d', '1m', '60d'
# Add config for number of emails retrieved
NUMBER_OF_EMAILS_RETRIEVED = 15 # Default number of emails for retriever

# --- Menu Tool Config ---
FLIK_SCHOOL_SUBDOMAIN = "cate"
FLIK_SCHOOL_IDENTIFIER = "cate-school-high-school"

# --- Web Scraper Tool Config ---
# (No specific config needed here currently, relies on hardcoded URL)

# --- Google Sheet Tool Config ---
# (No specific config needed here currently, relies on hardcoded Sheet ID)

# --- Streamlit Session Config ---
SESSION_ID = "streamlit_user_session"

# --- Date Config ---
TODAY_DATE = date.today().isoformat()