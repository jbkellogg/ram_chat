import os.path
import pickle
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from langchain.tools import Tool
import traceback
import config  # Import the config file

# --- Configuration ---
DOCS_CREDENTIALS_FILE = 'credentials.json'  # Use the main credentials file
DOCS_TOKEN_FILE = 'token.pickle'            # Use the main token file
END_OF_YEAR_DOC_ID = "1s43c4u0gRi7q6SQ8MAEGyDjHKcdqQiezttjbuXgyGxc"

# --- Authentication ---
def authenticate_docs():
    """Handles Google Docs API authentication using combined scopes."""
    creds = None
    if os.path.exists(DOCS_TOKEN_FILE):
        try:
            with open(DOCS_TOKEN_FILE, 'rb') as token:
                creds = pickle.load(token)
        except (EOFError, pickle.UnpicklingError):
            print(f"Error loading {DOCS_TOKEN_FILE}. Re-authenticating.")
            creds = None
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception as e:
                print(f"Error refreshing Docs token: {e}. Re-authenticating.")
                if os.path.exists(DOCS_TOKEN_FILE):
                    os.remove(DOCS_TOKEN_FILE)
                creds = None  # Force re-authentication
        else:
            if not os.path.exists(DOCS_CREDENTIALS_FILE):
                print(f"ERROR: Docs credentials file not found at {DOCS_CREDENTIALS_FILE}")
                return None
            # Use COMBINED_SCOPES from config
            flow = InstalledAppFlow.from_client_secrets_file(
                DOCS_CREDENTIALS_FILE, config.COMBINED_SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open(DOCS_TOKEN_FILE, 'wb') as token:
            pickle.dump(creds, token)
            print(f"Docs credentials saved to {DOCS_TOKEN_FILE}")

    try:
        service = build('docs', 'v1', credentials=creds)
        print("Google Docs API service created successfully (using combined scopes).")
        return service
    except HttpError as error:
        print(f'An error occurred building the Docs service: {error}')
        return None
    except Exception as e:
        print(f'An unexpected error occurred building Docs service: {e}')
        return None

# --- Helper Function to Read Structural Elements ---
def read_structural_elements(elements):
    """
    Recursively reads text from a list of structural elements
    (like paragraphs, tables) found within the document body or table cells.
    """
    text = ""
    for value in elements:
        if 'paragraph' in value:
            para_elements = value.get('paragraph').get('elements')
            for elem in para_elements:
                text_run = elem.get('textRun')
                if text_run:
                    text += text_run.get('content')
        elif 'table' in value:
            # Process table content
            table = value.get('table')
            for row in table.get('tableRows', []):
                row_text = []
                for cell in row.get('tableCells', []):
                    # Recursively read content within the cell
                    cell_text = read_structural_elements(cell.get('content', []))
                    row_text.append(cell_text.strip())
                # Join cell texts with a separator (e.g., | or tab) for readability
                text += "| " + " | ".join(row_text) + " |\n"
            text += "\n" # Add a newline after the table
        elif 'sectionBreak' in value:
            # Potentially add a separator for section breaks if needed
            text += "\n---\n"
        # Add handling for other element types like lists if necessary
        # elif ...

    return text

# --- Function to Read Document Content ---
def get_google_doc_content(doc_id: str) -> str:
    """Fetches and extracts text content from a Google Doc, including tables."""
    print(f"--- Attempting to fetch Google Doc content (ID: {doc_id}) ---")
    service = authenticate_docs()
    if not service:
        return "Error: Could not authenticate with Google Docs API."

    try:
        # Request the document content
        document = service.documents().get(documentId=doc_id).execute()
        doc_body = document.get('body')
        if not doc_body:
            return f"Error: Document (ID: {doc_id}) has no body content."

        doc_content = doc_body.get('content')
        if not doc_content:
            return f"Error: Document body (ID: {doc_id}) has no structural content."

        print("Extracting text from document structure (including tables)...")
        # Use the helper function to read all structural elements
        text_content = read_structural_elements(doc_content)

        if not text_content.strip():
             print(f"Warning: No text content extracted from Doc ID {doc_id}. Structure might be complex or unsupported.")
             return f"No readable text content found in Google Doc (ID: {doc_id})."

        print(f"Successfully extracted text content (length: {len(text_content)}).")
        # Return a formatted string
        title = document.get('title', 'End of Year Schedule Document')
        # Add a clear separator between title and content
        return f"Content from Google Doc '{title}' (ID: {doc_id}):\n{'-'*20}\n\n{text_content.strip()}"

    except HttpError as err:
        print(f"HTTP error fetching/parsing Google Doc: {err}")
        # Check for specific permission errors
        if err.resp.status == 403:
             reason = "Permission denied"
             try: # Try to get more details if available
                 details = err.resp.get('content', b'').decode()
                 if "ACCESS_TOKEN_SCOPE_INSUFFICIENT" in details:
                     reason = "Insufficient authentication scopes (token needs update?)"
                 elif "PERMISSION_DENIED" in details:
                     reason = "User lacks permission to access the document"
             except Exception: pass
             return f"Error accessing Google Doc (ID: {doc_id}): {reason}. Check permissions, scopes, and Doc ID."
        return f"Error accessing Google Doc (ID: {doc_id}): {err}. Check permissions and Doc ID."
    except Exception as e:
        print(f"Unexpected error fetching/parsing Google Doc: {e}")
        traceback.print_exc()
        return f"An unexpected error occurred while fetching the Google Doc: {e}"

# --- Tool Creation Function ---
def create_google_doc_tool():
    """Creates the Langchain Tool for fetching the end-of-year schedule doc."""
    # Wrapper to always use the specific doc ID
    def get_end_of_year_schedule_doc(input_str: str = "") -> str:
        """
        Tool wrapper that fetches content from the specific end-of-year schedule Google Doc.
        Ignores any input string.
        """
        # Input string is ignored, always fetch the specific document
        return get_google_doc_content(END_OF_YEAR_DOC_ID)

    doc_tool = Tool(
        name="get_end_of_year_schedule",
        func=get_end_of_year_schedule_doc,
        description=(
            "Use this tool ONLY for questions about the CATE SCHOOL END-OF-YEAR SCHEDULE, including review week, final exams, check-out, commencement, and related activities specifically for the period May 19th to May 31st, 2025. "
            "This tool retrieves information directly from the official 'End of Year Schedule 2025' Google Doc. "
            "Use this alongside `email_retriever` to check for any related updates or announcements in emails."
        ),
    )
    return doc_tool

# Example usage (optional)
if __name__ == '__main__':
    print("Testing Google Doc Tool...")
    content = get_google_doc_content(END_OF_YEAR_DOC_ID)
    print("\n--- Retrieved Content ---")
    print(content[:1000] + "..." if len(content) > 1000 else content)
    print("\n-----------------------")