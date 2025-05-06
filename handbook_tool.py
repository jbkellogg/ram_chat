import streamlit as st # Import Streamlit
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import Tool
from langchain_core.documents import Document # Import Document for type hinting

def create_handbook_tool(llm, retriever):
    """Creates the RAG tool for the handbook."""

    rag_prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:
<context>
{context}
</context>

Question: {input}""")
    rag_document_chain = create_stuff_documents_chain(llm, rag_prompt)
    # This chain returns {'input': ..., 'context': ..., 'answer': ...}
    rag_retrieval_chain = create_retrieval_chain(retriever, rag_document_chain)

    def invoke_rag_chain(input_dict: dict) -> str:
        """
        Invokes the RAG chain for handbook questions.
        Stores the retrieved context in session state and returns the answer.
        """
        st.session_state['last_used_context'] = None # Clear previous context first
        if isinstance(input_dict, dict) and "input" in input_dict:
            query = input_dict["input"]
        elif isinstance(input_dict, str):
            query = input_dict
        else:
            return "Error: Invalid input format for RAG tool."
        try:
            # Invoke the chain which returns context and answer
            response = rag_retrieval_chain.invoke({"input": query})
            answer = response.get("answer", "No answer found in context.")
            context: list[Document] | None = response.get("context")

            # Store the context in session state
            if context:
                st.session_state['last_used_context'] = context
                print(f"Stored {len(context)} handbook context documents in session state.") # Log for debugging

            return answer # Return only the answer string to the agent
        except Exception as e:
            print(f"Error invoking RAG chain: {e}") # Log error to console
            return f"Error invoking RAG chain: {e}"

    handbook_tool = Tool(
        name="cate_handbook_retriever",
        func=invoke_rag_chain,
        description="Use this tool to answer questions about the CATE School student handbook, rules, policies, and general school information.",
    )
    return handbook_tool