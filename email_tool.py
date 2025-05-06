import datetime
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import Tool
import streamlit as st
from langchain_core.documents import Document

# Note: We don't need email_downloader imports here, they are handled in vector_store.py

def create_email_tool(llm, retriever):
    """Creates the RAG tool for querying emails."""

    # --- System Prompt - REMOVED chat_history placeholder ---
    # Includes detailed instructions for date interpretation and tone
    system_prompt_text = f"""
    Answer the following question based only on the provided context (emails).

    You do not need to say things like "based on the provided context." That is implied.

    Today is {datetime.date.today()}. Use the 'timestamp' (Unix seconds) and 'date_str' metadata
    in the provided context to determine temporal relationships like "today", "tomorrow", and "this weekend."
    - If an email says "today", they mean the date on which the email was sent (infer from 'timestamp' or 'date_str'). Unless that date is today ({datetime.date.today()}), that event is not happening today.
    - If an email says "tomorrow", they mean the date after which the email was sent.
    - If an email says "this weekend", it refers to the upcoming Friday, Saturday, and Sunday relative to the email's date.
    - If an email says "Saturday", it refers to the next Saturday on the calendar relative to the email's date.
    - If an email says "Now", it refers to the instant corresponding to the 'timestamp' of the email.
    - If an email says "math lab is happening right now", that means it was happening when the email was sent, not in this present moment.
    In all of these cases, use the date information ('timestamp', 'date_str') associated with the email context to infer the date of the event.

    When you have conflicting information, prioritize more recent emails (higher 'timestamp').

    Avoid mixing information from separate emails unless they clearly refer to each other or the question implies combining information (e.g., "summarize emails about X").

    Before stating a specific date for an event mentioned (e.g., "The dance will be on Saturday"), verify that the date calculated based on the email's timestamp and the relative term used corresponds to the *next* occurrence of that day from today's perspective, if relevant to the user's question timing.

    Your tone should be happy, friendly, and informal.

<context>
{{context}}
</context>""" # Removed Chat History section

    # --- RAG Chain Setup ---
    # This prompt now only expects 'context' and 'input'
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt_text),
            ("human", "{input}"),
        ]
    )

    document_chain = create_stuff_documents_chain(llm, prompt)
    # Use the passed retriever (which should be the email retriever)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # --- Tool Definition ---
    # This wrapper directly invokes the retrieval chain for emails
    def invoke_email_rag_chain(input_dict: dict) -> str:
        """
        Invokes the RAG chain for email questions.
        Stores the retrieved context in session state and returns the answer.
        Prints retrieved context to console.
        """
        st.session_state['last_used_context'] = None
        if isinstance(input_dict, dict) and "input" in input_dict:
            query = input_dict["input"]
        elif isinstance(input_dict, str):
            query = input_dict
        else:
            return "Error: Invalid input format for Email RAG tool."
        try:
            # This chain returns {'input': ..., 'context': ..., 'answer': ...}
            response = retrieval_chain.invoke({"input": query})
            answer = response.get("answer", "Error: Could not generate answer from context.")
            context: list[Document] | None = response.get("context")

            # --- PRINT CONTEXT TO CONSOLE ---
            if context:
                print("\n--- Context Retrieved by Email Tool ---")
                for i, doc in enumerate(context):
                    print(f"  Document {i+1}:")
                    print(f"    Metadata: {doc.metadata}")
                    # Print first ~200 chars of content to keep it manageable
                    content_snippet = doc.page_content.replace('\n', ' ').strip()[:200]
                    print(f"    Content Snippet: {content_snippet}...")
                print("--- End Email Tool Context ---\n")
            else:
                print("\n--- No Context Retrieved by Email Tool ---\n")
            # --- END PRINT CONTEXT ---

            # Store the context in session state (existing code)
            if context:
                st.session_state['last_used_context'] = context
            else:
                st.session_state['last_used_context'] = None

            return answer
        except Exception as e:
            print(f"Error invoking Email RAG chain: {e}")
            return f"Error invoking Email RAG chain: {e}"

    email_tool = Tool(
        name="email_retriever",
        func=invoke_email_rag_chain, # Use the wrapper
        description="Use this tool ONLY to answer questions based on the content of emails. Useful for finding information about recent communications, schedules mentioned in emails, or specific details from past emails.",
    )
    return email_tool
