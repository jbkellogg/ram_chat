import datetime
from langchain.agents import AgentExecutor, create_react_agent, Tool 
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.query_constructor.base import AttributeInfo
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.documents import Document
import streamlit as st

import config
from vector_store import (
    get_embedding_model,
    get_handbook_vector_store, get_handbook_retriever,
    initialize_email_vector_store,
    get_email_standard_retriever,
    get_email_self_query_retriever
)
from handbook_tool import create_handbook_tool
from menu_tool import create_menu_tool 
from email_tool import create_email_tool
from web_scraper_tool import create_web_scraper_tool 
from google_sheet_tool import create_google_sheet_tool 
from google_doc_tool import create_google_doc_tool 

# --- History Setup ---
def get_streamlit_session_history(session_id: str) -> BaseChatMessageHistory:
    """
    Gets the chat history from Streamlit session state.
    If there is no history, create one.
    """
    if "lc_history" not in st.session_state:
        st.session_state.lc_history = ChatMessageHistory()
    return st.session_state.lc_history

# --- Get Last Context Tool (Modified Output Format) ---
def get_last_context(tool_input: str = "") -> str:
    """
    Retrieves and formats the context used for the last RAG answer.
    Returns a structured string listing sources.
    Ignores any input passed by the agent executor.
    """
    context: list[Document] | None = st.session_state.get('last_used_context')
    if not context:
        return "No specific context was stored from the last answer (it might have been a direct answer or from the menu tool)."

    # Use a more structured, simpler format less likely to cause parsing issues
    output_lines = ["Context used for the previous answer:"]
    for i, doc in enumerate(context):
        source_info = f"Source {i+1}: "
        metadata = doc.metadata
        if metadata:
            source = metadata.get('source')
            if source == 'gmail':
                sender = metadata.get('sender', 'N/A')
                subject = metadata.get('subject', 'N/A')
                date_str = metadata.get('date_str', 'N/A')
                source_info += f"Email | From: {sender} | Subject: '{subject}' | Date: {date_str}"
            elif source == config.PDF_DOC_PATH:
                 page = metadata.get('page', 'N/A')
                 source_info += f"Handbook PDF | Page: {page}"
            else: # Fallback
                 source_info += f"Unknown Source | Metadata: {metadata}"
        else:
            source_info += "No metadata available."
        output_lines.append(source_info)
        # Optionally add a very short snippet if needed, but keep it minimal
        # snippet = doc.page_content[:50].replace('\n', ' ') + "..."
        # output_lines.append(f"  Snippet: {snippet}")

    # Join lines into a single string for the observation
    return "\n".join(output_lines)

# --- Metadata Schema Definition for Self-Querying (if used) ---
email_metadata_field_info = [
    AttributeInfo(
        name="source",
        description="The source of the document, should always be 'gmail'",
        type="string",
    ),
    AttributeInfo(
        name="subject",
        description="The subject line of the email",
        type="string",
    ),
    AttributeInfo(
        name="timestamp",
        description=(
            "The date and time the email was received, stored as an integer Unix timestamp "
            "(seconds since epoch UTC). **Use this field EXCLUSIVELY for filtering or sorting emails by date.** "
            "For queries asking for the 'latest', 'newest', 'most recent', or 'next' information, you should filter for documents with a timestamp greater than a relevant past date or sort by this field in descending order. "
            "Example filter for emails after May 1, 2025 (UTC): {'comparator': 'gt', 'attribute': 'timestamp', 'value': 1746057600}. "
            "**Remember the value MUST be an integer timestamp.** Today's date is " + datetime.date.today().strftime('%Y-%m-%d') + "."
        ),
        type="integer",
    ),
    AttributeInfo(
        name="message_id",
        description="The unique identifier for the Gmail message",
        type="string",
    ),
]

# --- Agent Creation Function ---
def create_conversational_agent():
    """Initializes models, tools, and creates the conversational agent chain using ReAct."""
    print("--- Initializing Models and Embeddings ---")
    llm = ChatGoogleGenerativeAI(model=config.LLM_MODEL_NAME, temperature=0)
    embedding_model = get_embedding_model()

    print("--- Setting up Stores and Retrievers ---")
    print("--- Creating Handbook Retriever ---")
    handbook_vector_store = get_handbook_vector_store(embedding_model)
    handbook_retriever = get_handbook_retriever(handbook_vector_store)

    print("--- Creating Email Retriever (based on config) ---")
    email_vector_store = initialize_email_vector_store(embedding_model)
    if config.USE_EMAIL_SELF_QUERY:
        email_retriever = get_email_self_query_retriever(
            vector_store=email_vector_store,
            llm=llm,
            metadata_field_info=email_metadata_field_info,
        )
    else:
        email_retriever = get_email_standard_retriever(vector_store=email_vector_store)

    print("--- Creating Tools ---")
    try:
        handbook_tool = create_handbook_tool(llm, handbook_retriever)
        menu_tool = create_menu_tool() # Use the imported function
        email_tool = create_email_tool(llm, email_retriever)
        web_scraper_tool = create_web_scraper_tool()
        student_info_tool = create_google_sheet_tool()
        end_of_year_schedule_tool = create_google_doc_tool() # Instantiate the new tool
        get_last_context_tool = Tool(
            name="get_last_answer_context",
            func=get_last_context,
            description=(
                "Use this tool ONLY when the user explicitly asks for the sources, references, or context used for the immediately preceding answer. "
                "Provides identifying info and snippets. Present its output directly."
            ),
        )
        tools = [
            handbook_tool, menu_tool, email_tool, web_scraper_tool, 
            student_info_tool, end_of_year_schedule_tool, get_last_context_tool
        ] 

        # --- MODIFIED Tool Descriptions for ReAct ---
        email_tool.description = (
            "Use this tool for questions about school rules, policies, events, communications, and **potentially** athletics schedules/results mentioned in emails. "
            "**Input should be the user's natural language question** (e.g., 'emails from Ben Elkin', 'what was the schedule last Tuesday?'). "
            "**DO NOT format the input like a search query**. The tool handles filtering internally. "
            "Use this alongside `get_cate_athletics_info` for sports questions."
        )
        web_scraper_tool.description = ( # Matches the description in create_web_scraper_tool
            "Use this tool ONLY for questions about CATE SCHOOL ATHLETICS schedules, scores, results, or team information. "
            "This tool scrapes the official Cate athletics website (cate.org/athletics/...) and is the **primary source for schedules and results**. "
            "**Input MUST be ONLY the specific sport name string** (e.g., 'girls lacrosse', 'boys water polo', 'football'). Do NOT wrap it in a dictionary. "
            "Use this alongside `email_retriever` to check for related announcements."
        )
        student_info_tool.description = ( # Updated description
             "Use this tool ONLY to get the Birthday, Advisor, Mobile Phone, or Dorm for a specific CATE student. "
             "Input MUST be the student's full name (preferred for accuracy) or first name or last name. "
             "If multiple students match a first name, it will ask for clarification."
        )
        handbook_tool.description = (
             "Fallback tool if 'email_retriever' fails for specific CATE School rules/policies. Input should be the user's natural language question. Do NOT use for schedules/events/athletics."
        )
        menu_tool.description = (
            "Use this tool ONLY for questions about the food menu for a specific meal (breakfast, lunch, or dinner) on a specific date. "
            "**Input MUST be a JSON string** containing 'meal_type' (e.g., 'lunch') and 'date_str'. "
            "For 'date_str', use 'today', 'tomorrow', or a specific date in 'YYYY-MM-DD' format. "
            "Example Input: '{{\"meal_type\": \"lunch\", \"date_str\": \"2025-05-06\"}}' or '{{\"meal_type\": \"dinner\", \"date_str\": \"today\"}}'. "
            "Determine the meal type and the specific date. If the user says 'today' or 'tomorrow', use those exact words for 'date_str'. "
            "For all other dates (like 'Wednesday', 'next Friday', 'May 10th'), calculate the exact date in YYYY-MM-DD format and use that for 'date_str'. "
            "If the user asks a follow-up question about a menu (e.g., 'is it vegetarian?') and doesn't specify a date, check the chat history to see which date was discussed previously and use that date in the JSON input."
        )
        end_of_year_schedule_tool.description = (
            "Use this tool ONLY for questions about the CATE SCHOOL END-OF-YEAR SCHEDULE, including review week, final exams, check-out, commencement, and related activities specifically for the period May 19th to May 31st, 2025. "
            "This tool retrieves information directly from the official 'End of Year Schedule 2025' Google Doc. "
        )
        # --- END MODIFIED Tool Descriptions ---

    except Exception as e:
         st.error(f"An unexpected error occurred during tool creation: {e}")
         import traceback
         traceback.print_exc()
         st.stop()

    print("--- Defining ReAct Agent Prompt ---")
    custom_instructions = f"""You are a helpful assistant for CATE School. Today is {datetime.date.today().strftime('%Y-%m-%d')}.
Your primary goal is to answer questions based *only* on information retrieved from your tools, unless the question is simple conversational chit-chat.
You are also a ram. You make ram noises sometimes. Baaaah!

**CRITICAL INSTRUCTION: You MUST use your tools to answer informational questions. Do NOT rely on your internal knowledge or memory, even if you think you know the answer. If you think you know, you MUST use a tool to verify. This applies ESPECIALLY to schedules, results, policies, and future events.**

**Tool Usage Priority & Strategy:**
1.  **End-of-Year Schedule (May 19-31, 2025):** For questions about review week, exams, check-out, commencement during this specific period:
    *   **First, use `get_end_of_year_schedule`**. This accesses the official Google Doc.
    *   **Then, use `email_retriever`** with the user's question to check for any related updates or details mentioned in emails.
    *   Synthesize the information. Prioritize the Google Doc but mention relevant email updates.
2.  **get_dining_menu:** Use ONLY for food menu questions.
3.  **Athletics Questions (Schedules/Results):**
    *   **First, use `get_cate_athletics_info`** with the sport name.
    *   **Then, use `email_retriever`** to check for related announcements.
    *   Synthesize. Prioritize website, mention email changes.
4.  **Student Info (Birthday, Advisor, Mobile, Dorm):** Use `get_student_info` with the student's full name if possible.  First or last name is ok.
5.  **Other Informational Questions (Policies, Events outside May 19-31, Communications):** Use `email_retriever`.
6.  **cate_handbook_retriever:** Use ONLY as a fallback if `email_retriever` fails for specific official rules/policies.
7.  **get_last_answer_context:** Use ONLY if the user asks for the source/context for the previous answer.

**Answering Rules:**
- **ALWAYS use a tool** following the priority/strategy above for informational questions.
- **NEVER answer an informational question without first successfully invoking the appropriate tool(s) and getting a result.** Base your answer *strictly* on the tool's output.
- **When using `get_student_info`, clearly state the information found for the specific student.**
- **When a tool provides the requested information (like a schedule, menu, or student details), you MUST include the full details from the tool's observation directly in your final answer.** Do not just say the information is available or refer to the observation abstractly (e.g., don't just say "The schedule is available above").
- **Pay close attention to the chat history.** If the user asks a follow-up question that seems incomplete (e.g., missing a date or subject), use the history to understand the context and fill in the missing details for the tool input if possible.
- **Do not invent schedules, opponents, results, policies, or any other information.**A
- If tools fail or don't contain the info, state clearly that you could not find the information using your tools (e.g., "I checked the athletics website and recent emails, but couldn't find the specific result you asked for.").

TOOLS:
------
You have access to the following tools:"""

    react_template = f"""{custom_instructions}

{{tools}}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do. Use the tool priority/strategy. Check chat history for context if the question is a follow-up. Do not answer from memory. For end-of-year schedule, plan to use both the doc tool and email tool.
Action: the action to take, should be one of [{{tool_names}}]
Action Input: the input to the action. For `get_end_of_year_schedule`, the input is ignored. For others, ensure parameters are correct.
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer based *only* on the tool observations and the conversation history. I will synthesize information if multiple tools were used. **Crucially, I must include the actual schedule/menu/policy details directly from the relevant Observation(s) in my final answer.**
Final Answer: the final answer to the original input question, based strictly on tool observations. **Include all relevant details directly from the Observation.** Do not just say "The information is above" or similar. If tools failed, state that clearly.

Begin!

Previous conversation history:
{{chat_history}}

New input:
Question: {{input}}
Thought:{{agent_scratchpad}}"""

    react_prompt_template = PromptTemplate.from_template(react_template)

    print("--- Creating ReAct Agent and Executor ---")
    agent = create_react_agent(llm, tools, react_prompt_template)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=config.AGENT_IS_VERBOSE,
        handle_parsing_errors=True # Good practice for ReAct
    )

    print("--- Wrapping Agent with History ---")
    conversational_agent_chain = RunnableWithMessageHistory(
        agent_executor,
        get_streamlit_session_history,
        input_messages_key="input",
        history_messages_key="chat_history", # This key MUST match the placeholder in the template
        output_messages_key="output",
    )

    return conversational_agent_chain