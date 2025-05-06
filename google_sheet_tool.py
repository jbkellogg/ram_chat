import os.path
import pickle
import traceback
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from langchain.tools import StructuredTool
from pydantic.v1 import BaseModel, Field
import config # Assuming config stores paths/IDs
import re

# --- Configuration ---
SHEETS_SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']
SHEETS_CREDENTIALS_FILE = 'credentials.json'
SHEETS_TOKEN_FILE = 'token.pickle'
STUDENT_INFO_SHEET_ID = "1SjMyBVkYGaqn1MJwwIYZ-lWpMr8KBcUbgj3KyxTCHFU"

# --- Define expected column names ---
LAST_NAME_COLUMN = "Last Name"
FIRST_NAME_COLUMN = "First Name"
BIRTHDAY_COLUMN = "Birthday"
ADVISOR_COLUMN = "Advisor"
MOBILE_COLUMN = "Mobile Phone"
DORM_COLUMN = "Dorm"
EMAIL_COLUMN = "Email 1" # Added Email column

# --- Define the range to read ---
# Ensure this range includes all necessary columns (A=Last, B=First, H=Bday, I=Email, J=Mobile, F=Dorm)
READ_RANGE = "Master!A:K" # K covers Mobile Phone

# --- Authentication for Sheets ---
def authenticate_sheets():
    """Handles Google Sheets API authentication."""
    creds = None
    if os.path.exists(SHEETS_TOKEN_FILE):
        try:
            with open(SHEETS_TOKEN_FILE, 'rb') as token:
                creds = pickle.load(token)
        except (EOFError, pickle.UnpicklingError):
            print(f"Error loading {SHEETS_TOKEN_FILE}. Re-authenticating.")
            creds = None

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception as e:
                print(f"Error refreshing Sheets token: {e}. Re-authenticating.")
                if os.path.exists(SHEETS_TOKEN_FILE):
                    os.remove(SHEETS_TOKEN_FILE)
                creds = None
        else:
            if not os.path.exists(SHEETS_CREDENTIALS_FILE):
                print(f"Error: Sheets credentials file '{SHEETS_CREDENTIALS_FILE}' not found.")
                return None
            flow = InstalledAppFlow.from_client_secrets_file(
                SHEETS_CREDENTIALS_FILE, config.COMBINED_SCOPES)
            creds = flow.run_local_server(port=0)

        with open(SHEETS_TOKEN_FILE, 'wb') as token:
            pickle.dump(creds, token)
            print(f"Sheets credentials saved to {SHEETS_TOKEN_FILE}")

    try:
        service = build('sheets', 'v4', credentials=creds)
        print("Google Sheets API service created successfully")
        return service
    except HttpError as error:
        print(f'An error occurred building the Sheets service: {error}')
        return None
    except Exception as e:
        print(f'An unexpected error occurred during Sheets service build: {e}')
        return None

# --- Pydantic Model for Input ---
class StudentInfoInput(BaseModel):
    student_name: str = Field(description="The full name (preferred) or first name of the student to look up.")

# --- Core Function to Get Student Data ---
def get_student_info(student_name: str) -> str:
    """
    Fetches specific information (Birthday, Advisor, Mobile, Dorm) for a given student
    from the predefined Google Sheet. Searches by full name (preferred) or first name.
    As a fallback, attempts to match the input name against the prefix of the student's email address.
    """
    print(f"Attempting to get info for student: '{student_name}' from Sheet ID: {STUDENT_INFO_SHEET_ID}")
    service = authenticate_sheets()
    if not service:
        return "Error: Failed to authenticate with Google Sheets API."

    try:
        sheet = service.spreadsheets()
        result = sheet.values().get(spreadsheetId=STUDENT_INFO_SHEET_ID,
                                    range=READ_RANGE).execute()
        values = result.get('values', [])

        if not values or len(values) < 2:
            return f"No data found or sheet is empty in range '{READ_RANGE}' of sheet '{STUDENT_INFO_SHEET_ID}'."

        header = values[0]
        data_rows = values[1:]
        print(f"DEBUG: Header row read from sheet: {header}") # Keep for debugging

        # --- Find column indices ---
        try:
            last_name_col_idx = header.index(LAST_NAME_COLUMN)
            first_name_col_idx = header.index(FIRST_NAME_COLUMN)
            bday_col_idx = header.index(BIRTHDAY_COLUMN)
            advisor_col_idx = header.index(ADVISOR_COLUMN)
            mobile_col_idx = header.index(MOBILE_COLUMN)
            dorm_col_idx = header.index(DORM_COLUMN)
            email_col_idx = header.index(EMAIL_COLUMN) # Get email index
        except ValueError as e:
            return f"Error: Missing expected column in sheet header: {e}. Header checked: {header}. Expected columns: ..., {EMAIL_COLUMN}"

        # --- Primary Search: By Name ---
        student_name_lower = student_name.lower().strip()
        is_likely_full_name = ' ' in student_name_lower or ',' in student_name_lower
        found_students = []

        print(f"DEBUG: Starting primary search for '{student_name_lower}'...")
        for row_idx, row in enumerate(data_rows):
            if len(row) <= max(first_name_col_idx, last_name_col_idx): continue

            first_name = row[first_name_col_idx].strip().lower()
            last_name = row[last_name_col_idx].strip().lower()
            full_name_sheet = f"{first_name} {last_name}"
            full_name_sheet_rev = f"{last_name}, {first_name}"
            full_name_sheet_lastfirst = f"{last_name} {first_name}"

            match = False
            if is_likely_full_name:
                if student_name_lower in [full_name_sheet, full_name_sheet_rev, full_name_sheet_lastfirst]:
                    match = True
            else: # Match only against first name
                if first_name == student_name_lower:
                    match = True

            if match:
                 print(f"DEBUG: Primary match found at row {row_idx+2}")
                 found_students.append({'row': row, 'name': f"{row[first_name_col_idx].strip()} {row[last_name_col_idx].strip()}"})

        # --- Fallback Search: By Email Prefix (if no primary match) ---
        if not found_students:
            print(f"DEBUG: Primary search failed. Starting fallback email prefix search for '{student_name_lower}'...")
            
            # Split the input name into parts (handle potential comma)
            input_name_parts = student_name_lower.replace(',', '').split()
            
            for row_idx, row in enumerate(data_rows):
                 if len(row) <= email_col_idx: continue
                 email = row[email_col_idx].strip().lower()
                 if not email or '@' not in email: continue

                 email_prefix = email.split('@')[0]
                 # Split prefix by common delimiters like '.' and '_'
                 email_prefix_parts = re.split(r'[._]', email_prefix) # Use regex to split by . or _

                 # Check if all parts of the input name exist in the email prefix parts
                 all_input_parts_found = True
                 for name_part in input_name_parts:
                     if name_part not in email_prefix_parts:
                         all_input_parts_found = False
                         break
                 
                 if all_input_parts_found:
                     # Check if this student was already found to avoid duplicates
                     already_found = any(s['row'] == row for s in found_students)
                     if not already_found:
                         print(f"DEBUG: Fallback email prefix match found at row {row_idx+2} (Email: {email})")
                         found_students.append({'row': row, 'name': f"{row[first_name_col_idx].strip()} {row[last_name_col_idx].strip()}"})

        # --- Process Results ---
        if not found_students:
            # Updated message
            return f"Could not find any student matching '{student_name}' by name or email prefix in the sheet."

        if len(found_students) > 1:
            matched_names = list(set(s['name'] for s in found_students)) # Use set to remove duplicates if any
            return (f"Found multiple potential matches for '{student_name}': {', '.join(matched_names)}. "
                    "Please provide a more specific full name for a unique result.")

        # Exactly one match found
        found_student_row = found_students[0]['row']
        found_student_name = found_students[0]['name']

        # --- Extract data safely ---
        def get_cell_value(row, index):
            return row[index].strip() if len(row) > index and row[index] else "N/A"

        birthday = get_cell_value(found_student_row, bday_col_idx)
        advisor = get_cell_value(found_student_row, advisor_col_idx)
        mobile = get_cell_value(found_student_row, mobile_col_idx)
        dorm = get_cell_value(found_student_row, dorm_col_idx)

        # --- Format output ---
        output = (
            f"Information for {found_student_name}:\n"
            f"- Birthday: {birthday}\n"
            f"- Advisor: {advisor}\n"
            f"- Mobile Phone: {mobile}\n"
            f"- Dorm: {dorm}"
        )
        return output

    except HttpError as err:
        print(f"Google Sheets API error: {err}")
        if err.resp.status == 404:
            return f"Error: Google Sheet not found (ID: {STUDENT_INFO_SHEET_ID})."
        elif err.resp.status == 403:
             return f"Error: Permission denied for sheet '{STUDENT_INFO_SHEET_ID}'."
        else:
             return f"Error accessing Google Sheet: {err}"
    except Exception as e:
        traceback.print_exc()
        return f"An unexpected error occurred fetching student info: {e}"

# --- Tool Creation Function ---
def create_google_sheet_tool():
    """Creates the Langchain Tool for reading specific student info from Google Sheet."""
    google_sheet_reader_tool = StructuredTool.from_function(
        func=get_student_info,
        name="get_student_info",
        description=( # Updated description
            "Use this tool ONLY to get the Birthday, Advisor, Mobile Phone, or Dorm for a specific CATE student. "
            "Input MUST be the student's full name (preferred for accuracy) or first name. "
            "The tool will first search by name, and if no match is found, it will attempt to match the input first name against the student's email address prefix as a fallback. "
            "If multiple students match, it will ask for clarification."
        ),
        args_schema=StudentInfoInput,
    )
    return google_sheet_reader_tool

# --- Example Usage ---
if __name__ == '__main__':
    # Test with full name
    test_student_full = "Kellogg, Jack" 
    print(f"\n--- Testing Google Sheet Tool (Full Name: {test_student_full}) ---")
    result_full = get_student_info(student_name=test_student_full)
    print("\n--- Result ---")
    print(result_full)

    # Test with first name (replace 'Jack' if needed)
    test_student_first = "Jack" 
    print(f"\n--- Testing Google Sheet Tool (First Name: {test_student_first}) ---")
    result_first = get_student_info(student_name=test_student_first)
    print("\n--- Result ---")
    print(result_first)