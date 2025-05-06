import os.path
import base64
import time
import pickle
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import requests
from langchain_core.documents import Document
from email.utils import parsedate_to_datetime  # For parsing RFC 2822 date strings
from datetime import datetime, timedelta  # For example usage
import config  # Import the config file

# If modifying these SCOPES, delete the file token.pickle.
CREDENTIALS_FILE = 'credentials.json'  # Make sure this matches your actual file
TOKEN_FILE = 'token.pickle'  # Use the same token file name everywhere


def authenticate_gmail():
    """Handles user authentication using combined scopes."""
    creds = None
    if os.path.exists(TOKEN_FILE):
        try:
            with open(TOKEN_FILE, 'rb') as token:
                creds = pickle.load(token)
        except (EOFError, pickle.UnpicklingError):
            print(f"Error loading {TOKEN_FILE}. Re-authenticating.")
            creds = None
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception as e:
                print(f"Error refreshing token: {e}. Re-authenticating.")
                if os.path.exists(TOKEN_FILE):
                    os.remove(TOKEN_FILE)
                creds = None
        else:
            if not os.path.exists(CREDENTIALS_FILE):
                print(f"Error: {CREDENTIALS_FILE} not found.")
                return None
            # Use COMBINED_SCOPES from config
            flow = InstalledAppFlow.from_client_secrets_file(
                CREDENTIALS_FILE, config.COMBINED_SCOPES)
            creds = flow.run_local_server(port=0)
        with open(TOKEN_FILE, 'wb') as token:
            pickle.dump(creds, token)
            print(f"Credentials saved to {TOKEN_FILE}")
    try:
        service = build('gmail', 'v1', credentials=creds)
        print("Gmail API service created successfully (using combined scopes)")
        return service
    except HttpError as error:
        print(f'An error occurred building the service: {error}')
        return None
    except Exception as e:
        print(f'An unexpected error occurred: {e}')
        return None


def list_messages(service, user_id='me', query=''):
    """Lists the user's messages matching the query."""
    try:
        response = service.users().messages().list(userId=user_id, 
                                                   q=query, 
                                                   ).execute()
        messages = []
        if 'messages' in response:
            messages.extend(response['messages'])
        while 'nextPageToken' in response:
            page_token = response['nextPageToken']
            response = service.users().messages().list(userId=user_id, 
                                                       q=query,
                                                       pageToken=page_token, 
                                                       ).execute()
            if 'messages' in response:
                messages.extend(response['messages'])
            else:
                break
        print(f"Found {len(messages)} messages matching query '{query}'.")
        return messages
    except HttpError as error:
        print(f'An error occurred listing messages: {error}')
        return None
    except Exception as e:
        print(f'An unexpected error occurred: {e}')
        return None


def get_message(service, msg_id, user_id='me', format='full'):
    """Gets the specified message."""
    try:
        message = service.users().messages().get(userId=user_id, id=msg_id, format=format).execute()
        return message
    except HttpError as error:
        print(f'An error occurred getting message {msg_id}: {error}')
        return None
    except Exception as e:
        print(f'An unexpected error occurred: {e}')
        return None


def parse_message_parts(parts):
    """Recursively parses message parts to find the text/plain body."""
    body = ""
    if parts:
        for part in parts:
            mimeType = part.get('mimeType')
            part_body = part.get('body')
            data = part_body.get('data')
            if mimeType == 'text/plain' and data:
                try:
                    decoded_data = base64.urlsafe_b64decode(data + '===').decode('utf-8', errors='replace')
                    return decoded_data
                except Exception as decode_error:
                    print(f"    Error decoding text/plain part: {decode_error}")
                    return "[Decoding Error]"
            elif 'parts' in part:
                found_body = parse_message_parts(part['parts'])
                if found_body:
                    return found_body
    return body


def fetch_emails_for_embedding(service, query='', max_results=None):
    """
    Fetches emails matching a query and structures them, including sender, date, subject, and body.
    Returns: A list of dictionaries, where each dictionary represents an email.
    """
    print(f"\n--- Fetching emails (query='{query}', max_results={max_results}) ---")
    structured_emails = []
    try:
        messages_info = list_messages(service, query=query)
        if not messages_info:
            print("No messages found matching the query.")
            return structured_emails

        if max_results is not None:
            messages_to_process = messages_info[:max_results]
            print(f"Processing up to {len(messages_to_process)} messages due to limit.")
        else:
            messages_to_process = messages_info
            print(f"Processing all {len(messages_to_process)} found messages.")

        for i, message_info in enumerate(messages_to_process):
            msg_id = message_info['id']
            print(f"  Processing message {i+1}/{len(messages_to_process)}, ID: {msg_id}")
            message = get_message(service, msg_id, format='full')
            if not message:
                print(f"    Could not retrieve message {msg_id}. Skipping.")
                continue
            payload = message.get('payload')
            if not payload:
                print(f"    Message {msg_id} has no payload. Skipping.")
                continue
            headers = payload.get('headers', [])
            subject, sender, date = "", "", ""
            for header in headers:
                name = header.get('name', '').lower()
                if name == 'subject':
                    subject = header.get('value', '')
                elif name == 'from':
                    sender = header.get('value', '')
                elif name == 'date':
                    date = header.get('value', '')

            body = ""
            if 'parts' in payload:
                body = parse_message_parts(payload.get('parts'))
            elif 'body' in payload and payload.get('body', {}).get('data'):
                data = payload.get('body', {}).get('data')
                try:
                    body = base64.urlsafe_b64decode(data + '===').decode('utf-8', errors='replace')
                except Exception as decode_error:
                    print(f"    Error decoding body for message {msg_id}: {decode_error}")
                    body = "[Decoding Error]"
            else:
                print(f"    Could not find body content for message {msg_id}.")
                body = "[No Body Found]"

            structured_emails.append({
                'id': msg_id, 'sender': sender, 'date': date,
                'subject': subject, 'body': body.strip()
            })
        print(f"--- Finished fetching {len(structured_emails)} emails ---")
        return structured_emails
    except HttpError as error:
        print(f'An error occurred fetching emails: {error}')
        return []
    except Exception as e:
        print(f'An unexpected error occurred during fetch: {e}')
        import traceback
        traceback.print_exc()
        return []


def create_langchain_documents_from_emails(email_data_list):
    """
    Converts a list of email data dictionaries into Langchain Document objects.

    Args:
        email_data_list: A list of dictionaries, as returned by fetch_emails_for_embedding.

    Returns:
        A tuple containing:
        - documents (list): A list of Langchain Document objects.
        - doc_ids (list): A list of corresponding message IDs for Chroma.
    """
    print(f"\n--- Creating Langchain Documents from {len(email_data_list)} emails ---")
    documents = []
    doc_ids = []
    skipped_count = 0

    for email_data in email_data_list:
        # --- Parse date string to timestamp ---
        timestamp = None
        try:
            dt_obj = parsedate_to_datetime(email_data['date'])
            if dt_obj:
                timestamp = int(dt_obj.timestamp())
        except Exception as date_parse_error:
            print(f"Warning: Could not parse date '{email_data['date']}' for msg {email_data['id']}: {date_parse_error}")
        # --- End date parsing ---
        print(email_data['date'])
        # Create the Document object
        doc = Document(page_content=f"""
            'sender': {email_data['sender']},
            'date_str': {email_data['date']},
            'subject': {email_data['subject']},
            'timestamp': {timestamp}
            'body':{email_data['body']}
            """,
        metadata={
            'source': 'gmail',
            'message_id': email_data['id'],
            'sender': email_data['sender'],
            'date_str': email_data['date'],
            'subject': email_data['subject'],
            'timestamp': timestamp
        }
        )   

        # Basic validation: Ensure there's content and a valid ID
        if doc.page_content and email_data['id']:
            documents.append(doc)
            doc_ids.append(email_data['id'])  # Use the email's unique ID for Chroma
        else:
            skipped_count += 1
            print(f"Skipping email ID {email_data.get('id', 'N/A')} due to missing content or ID.")

    print(f"Created {len(documents)} Langchain Documents. Skipped {skipped_count}.")
    return documents, doc_ids


if __name__ == '__main__':
    gmail_service = authenticate_gmail()

    if gmail_service:
        # --- Example: Fetch emails from the last 7 days ---
        seven_days_ago = (datetime.now() - timedelta(days=7)).strftime('%Y/%m/%d')
        query_last_week = f'after:{seven_days_ago}'
        emails_data = fetch_emails_for_embedding(gmail_service, query=query_last_week, max_results=10)

        if emails_data:
            # --- Convert fetched data to Langchain Documents ---
            langchain_documents, document_ids = create_langchain_documents_from_emails(emails_data)

            if langchain_documents:
                print("\n--- Example Langchain Document ---")
                print(langchain_documents[0])
                print("---------------------------------")
                print(f"Document ID: {document_ids[0]}")
                print("---------------------------------")

                # --- Add to Chroma (Example) ---
                # from langchain_chroma import Chroma
                # from langchain_google_genai import GoogleGenerativeAIEmbeddings
                #
                # try:
                #     embedding_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004") # Replace with your model
                #     persist_directory = "./gmail_chroma_db" # Choose a directory
                #
                #     # Option 1: Create or load and add/update
                #     vector_store = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)
                #     vector_store.add_documents(documents=langchain_documents, ids=document_ids)
                #     print(f"\nAdded/Updated {len(langchain_documents)} documents in Chroma DB at {persist_directory}")
                #
                #     # Option 2: Create from scratch (if sure it's the first time)
                #     # vector_store = Chroma.from_documents(
                #     #     documents=langchain_documents,
                #     #     embedding=embedding_model,
                #     #     ids=document_ids,
                #     #     persist_directory=persist_directory
                #     # )
                #     # print(f"\nCreated Chroma DB at {persist_directory} with {len(langchain_documents)} documents.")
                #
                # except Exception as chroma_error:
                #      print(f"\nError interacting with Chroma: {chroma_error}")

    else:
        print("Could not authenticate or build Gmail service. Exiting.")
