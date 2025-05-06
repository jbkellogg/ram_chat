import os
import shutil
import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings
try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma # Fallback if needed

# Imports needed for Self Query
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo, Comparator # Import Comparator
from langchain_core.language_models import BaseLanguageModel # For type hinting LLM

import config # Import configuration
from datetime import datetime, timedelta
# Import email downloader functions (assuming they are in a sibling directory)
try:
    from email_downloader import authenticate_gmail, fetch_emails_for_embedding, create_langchain_documents_from_emails
except ImportError:
    print("Warning: Could not import email_downloader. Email updates/rebuilds will fail.")
    authenticate_gmail = None
    fetch_emails_for_embedding = None
    create_langchain_documents_from_emails = None


# --- Embedding Model ---
def get_embedding_model():
    """Initializes and returns the embedding model."""
    return GoogleGenerativeAIEmbeddings(model=config.EMBEDDING_MODEL_NAME)

# --- Handbook Vector Store ---
def get_handbook_vector_store(embedding_model):
    """Loads or creates the handbook vector store."""
    # Use the correct variable name from config.py
    persist_dir = config.HANDBOOK_DB_DIRECTORY # Changed from HANDBOOK_PERSIST_DIRECTORY
    if not os.path.exists(persist_dir) or config.REBUILD_HANDBOOK_VECTOR_STORE:
        print(f"Creating new handbook vector store in {persist_dir}...")
        if not os.path.exists(config.PDF_DOC_PATH):
             st.error(f"PDF file not found at {config.PDF_DOC_PATH}. Cannot create handbook vector store.")
             st.stop()

        print(f"Loading document: {config.PDF_DOC_PATH}")
        loader = PyPDFLoader(config.PDF_DOC_PATH)
        documents = loader.load()
        print(f"Loaded {len(documents)} pages.")

        print(f"Splitting documents into chunks (size={config.CHUNK_SIZE}, overlap={config.CHUNK_OVERLAP})...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP
        )
        chunks = text_splitter.split_documents(documents)
        print(f"Created {len(chunks)} chunks.")

        print("Generating embeddings and creating handbook vector store...")
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory=persist_dir
        )
        print("Handbook vector store created and persisted.")
    else:
        print(f"Loading existing handbook vector store from {persist_dir}...")
        vector_store = Chroma(
            persist_directory=persist_dir,
            embedding_function=embedding_model
        )
        print("Handbook vector store loaded.")
    return vector_store

# --- Handbook Retriever ---
def get_handbook_retriever(vector_store):
    """Creates a retriever for the handbook vector store."""
    return vector_store.as_retriever()

# --- Email Vector Store Initialization ---
def initialize_email_vector_store(embedding_model):
    """Loads or creates the vector store for emails, handling rebuilds and updates."""
    # Use the correct variable name from config.py
    persist_dir = config.EMAIL_DB_DIRECTORY # Changed from EMAIL_PERSIST_DIRECTORY
    vector_store = None

    if config.REBUILD_EMAIL_VECTOR_STORE:
        print(f"Rebuilding email vector store in {persist_dir}...")
        if os.path.exists(persist_dir):
            shutil.rmtree(persist_dir)
            print("  Old email vector store deleted.")
        os.makedirs(persist_dir, exist_ok=True)

        if authenticate_gmail and fetch_emails_for_embedding and create_langchain_documents_from_emails:
            gmail_service = authenticate_gmail()
            if gmail_service:
                query = f"newer_than:{config.TIME_FRAME_FOR_EMAIL_VECTOR_STORE_BUILD}"
                print(f"  Fetching emails for rebuild (query: '{query}')...")
                emails_data = fetch_emails_for_embedding(gmail_service, query=query)
                if emails_data:
                    langchain_documents, document_ids = create_langchain_documents_from_emails(emails_data)
                    if langchain_documents:
                        print(f"  Creating Chroma store from {len(langchain_documents)} documents...")
                        vector_store = Chroma.from_documents(
                            documents=langchain_documents,
                            embedding=embedding_model,
                            ids=document_ids,
                            persist_directory=persist_dir
                        )
                        print("  New email vector store created and persisted.")
                    else:
                        print("  No valid documents created from fetched emails. Store is empty.")
                        # Create an empty store to avoid errors later
                        vector_store = Chroma(persist_directory=persist_dir, embedding_function=embedding_model)
                else:
                    print("  No emails found for rebuild. Store is empty.")
                    vector_store = Chroma(persist_directory=persist_dir, embedding_function=embedding_model)
            else:
                print("  Gmail authentication failed. Cannot rebuild store.")
                st.error("Gmail authentication failed. Cannot rebuild email store.")
                st.stop()
        else:
            print("  Email downloader functions not available. Cannot rebuild store.")
            st.error("Email downloader functions not available. Cannot rebuild email store.")
            st.stop()

    else: # Load existing store or create if it doesn't exist
        if os.path.exists(persist_dir):
            print(f"Loading existing email vector store from {persist_dir}...")
            try:
                vector_store = Chroma(
                    persist_directory=persist_dir,
                    embedding_function=embedding_model
                )
                print("  Email vector store loaded.")
            except Exception as e:
                st.error(f"Error loading email vector store from {persist_dir}: {e}. Try rebuilding.")
                print(f"Error loading email vector store: {e}")
                st.stop()
        else:
            print(f"Email vector store not found at {persist_dir}. Creating empty store.")
            print("Run with REBUILD_EMAIL_VECTOR_STORE=True to populate it.")
            os.makedirs(persist_dir, exist_ok=True)
            vector_store = Chroma(persist_directory=persist_dir, embedding_function=embedding_model)
            st.warning(f"Email vector store not found. Created empty store at {persist_dir}. Re-run with rebuild flag if needed.")

    # --- Optional Update ---
    if not config.REBUILD_EMAIL_VECTOR_STORE and config.UPDATE_EMAIL_ON_STARTUP:
        print("Checking for email updates...")
        if authenticate_gmail and fetch_emails_for_embedding and create_langchain_documents_from_emails:
            gmail_service = authenticate_gmail()
            if gmail_service:
                # --- Use EMAIL_UPDATE_WINDOW_DAYS to calculate date ---
                try:
                    days_to_check = int(config.EMAIL_UPDATE_WINDOW_DAYS)
                    update_since_date = (datetime.now() - timedelta(days=days_to_check)).strftime('%Y/%m/%d')
                    query = f'after:{update_since_date}' # Use 'after:' format for Gmail API
                    print(f"  Fetching recent emails for update (query: '{query}')...")
                except (AttributeError, ValueError, TypeError) as e:
                     print(f"  Error processing EMAIL_UPDATE_WINDOW_DAYS from config: {e}. Skipping update.")
                     st.warning(f"Config error for EMAIL_UPDATE_WINDOW_DAYS: {e}. Skipping email update.")
                     query = None # Prevent fetching if config is bad

                if query: # Only proceed if query was constructed successfully
                    emails_data = fetch_emails_for_embedding(gmail_service, query=query)
                    if emails_data:
                        langchain_documents, document_ids = create_langchain_documents_from_emails(emails_data)
                        if langchain_documents:
                            print(f"  Adding/Updating {len(langchain_documents)} documents in the store...")
                            try:
                                vector_store.add_documents(documents=langchain_documents, ids=document_ids)
                                print("  Email vector store updated.")
                            except Exception as e:
                                 print(f"  Error adding documents during update: {e}")
                                 st.warning(f"Error updating email store: {e}")
                        else:
                            print("  No valid documents created from fetched emails for update.")
                    else:
                        print("  No new emails found for update.")
                # --- End date calculation and query ---
            else:
                print("  Gmail authentication failed. Cannot update store.")
                st.warning("Gmail authentication failed. Cannot update email store.")
        else:
            print("  Email downloader functions not available. Cannot update store.")
            st.warning("Email downloader functions not available. Cannot update email store.")

    # --- Optional Metadata Dump ---
    if config.DUMP_EMAIL_METADATA_ON_STARTUP:
        print("\n--- Dumping Email Metadata from Vector Store ---")
        if vector_store:
            try:
                # Fetch all entries (use include=['metadatas'] for efficiency)
                # Warning: This can be slow/memory-intensive for very large databases!
                results = vector_store.get(include=['metadatas'])
                all_metadata = results.get('metadatas', [])
                if all_metadata:
                    print(f"Found metadata for {len(all_metadata)} entries:")
                    for i, meta in enumerate(all_metadata):
                        print(f"  Entry {i+1}: {meta}")
                else:
                    print("  No metadata found in the vector store.")
            except Exception as e:
                print(f"  Error retrieving metadata from Chroma: {e}")
        else:
            print("  Vector store object is not available. Cannot dump metadata.")
        print("--- Finished Dumping Metadata ---\n")
    # --- End Metadata Dump ---


    if vector_store is None:
        st.error("Failed to load or create the email vector store.")
        st.stop()

    return vector_store


# --- Email Retriever Functions ---

def get_email_standard_retriever(vector_store: Chroma):
    """Creates a standard retriever for the email vector store."""
    print("Using standard email retriever.")
    k_value = config.NUMBER_OF_EMAILS_RETRIEVED # Use config value
    print(f"Standard retriever will fetch top {k_value} documents.")
    return vector_store.as_retriever(search_kwargs={"k": k_value})

def get_email_self_query_retriever(
    vector_store: Chroma,
    llm: BaseLanguageModel,
    metadata_field_info: list[AttributeInfo]
):
    """Creates a SelfQueryRetriever for the email vector store."""
    print("Using SelfQueryRetriever for emails.")
    document_content_description = "Content of an email message"
    k_value = config.NUMBER_OF_EMAILS_RETRIEVED # Get k from config
    print(f"SelfQueryRetriever will aim to retrieve top {k_value} documents.") # Add log
    try:
        retriever = SelfQueryRetriever.from_llm(
            llm=llm,
            vectorstore=vector_store,
            document_contents=document_content_description,
            metadata_field_info=metadata_field_info,
            verbose=config.AGENT_IS_VERBOSE,
            allowed_comparators=[ # Explicitly allow comparators
                Comparator.EQ, Comparator.NE, Comparator.GT,
                Comparator.GTE, Comparator.LT, Comparator.LTE, Comparator.LIKE
            ],
            # --- Pass search_kwargs to the underlying retriever ---
            search_kwargs={"k": k_value}
            # --- End passing search_kwargs ---
        )
        return retriever
    except Exception as e:
        print(f"Error creating SelfQueryRetriever: {e}")
        print("Falling back to standard email retriever.")
        # Ensure fallback also uses the correct k
        return get_email_standard_retriever(vector_store)

# --- Standalone Email Update Function ---
def update_email_vector_store_manual(persist_dir: str, embedding_model, days_to_check: int = 7) -> tuple[bool, str]:
    """
    Checks for new emails within the specified window and adds them to the existing vector store.
    """
    print(f"--- Manual Email Update Triggered (Checking last {days_to_check} days) ---")
    if not os.path.exists(persist_dir):
        return False, f"Error: Vector store directory not found at {persist_dir}. Cannot update."

    try:
        # Load the existing store
        print(f"  Loading existing store from {persist_dir} for update...")
        vector_store = Chroma(
            persist_directory=persist_dir,
            embedding_function=embedding_model
        )

        # Calculate date range and query
        update_since_date = (datetime.now() - timedelta(days=days_to_check)).strftime('%Y/%m/%d')
        update_query = f'after:{update_since_date}'
        print(f"  Querying Gmail for emails {update_query}...")

        gmail_service = authenticate_gmail()
        if not gmail_service:
            return False, "Update failed: Gmail authentication failed."

        emails_data = fetch_emails_for_embedding(gmail_service, query=update_query)
        if not emails_data:
            return True, f"No new emails found since {update_since_date}."

        print(f"  Found {len(emails_data)} potential new emails.")
        new_documents, new_doc_ids = create_langchain_documents_from_emails(emails_data)

        if not new_documents:
            return True, "Fetched emails, but no valid new documents were created to add."

        print(f"  Adding/updating {len(new_documents)} documents in the vector store...")
        # Add documents - persistence should be handled automatically by Chroma when loaded this way
        vector_store.add_documents(documents=new_documents, ids=new_doc_ids)

        print("  Vector store updated.") # Adjusted message
        return True, f"Successfully added/updated {len(new_documents)} email documents."

    except Exception as update_err:
         print(f"  Error during manual email update: {update_err}")
         import traceback
         traceback.print_exc()
         return False, f"An error occurred during the update: {update_err}"
# --- End Standalone Email Update Function ---