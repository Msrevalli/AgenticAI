import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain.tools.retriever import create_retriever_tool
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain import hub
from typing_extensions import TypedDict
from typing import Annotated, Sequence
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Streamlit UI
st.set_page_config(page_title="Chat with Documents", layout="wide")
st.title("📚 Chat with Your Documents (Agentic RAG)")

# Initialize session state for conversation
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Input type selection
input_option = st.radio("Select Input Type:", ["Upload PDF", "Enter URL"])

uploaded_files = []
urls = []

if input_option == "Upload PDF":
    uploaded_files = st.file_uploader("Upload PDFs", accept_multiple_files=True, type=["pdf"])

if input_option == "Enter URL":
    urls = st.text_area("Enter URLs (one per line)").split("\n")

# Load documents
documents = []
if uploaded_files:
    for uploaded_file in uploaded_files:
        file_path = os.path.join("temp_files", uploaded_file.name)
        os.makedirs("temp_files", exist_ok=True)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Load PDF
        loader = PyPDFLoader(file_path)
        documents.extend(loader.load())

        # Optional: Delete file after processing
        os.remove(file_path)

if urls:
    documents.extend([WebBaseLoader(url).load()[0] for url in urls])

# Process documents if available
if documents:
    st.success(f"Loaded {len(documents)} documents.")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    doc_splits = text_splitter.split_documents(documents)

    # Create FAISS vector database
    vectorstore = FAISS.from_documents(doc_splits, embedding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()
    
    # Create retriever tool
    retriever_tool = create_retriever_tool(
        retriever,
        "retriever_vector_db",
        "Retrieve relevant information from documents"
    )
    
    tools = [retriever_tool]

    # Define Agent State
    class AgentState(TypedDict):
        messages: Annotated[Sequence[HumanMessage | AIMessage], None]

    # Define Agent Function
    def agent(state):
        """Decides whether to retrieve, rewrite, or generate response."""
        messages = state["messages"]
        model = ChatGroq(model="qwen-2.5-32b").bind_tools(tools)
        response = model.invoke(messages)
        return {"messages": messages + [response]}

    # Define Query Rewriting
    def rewrite(state):
        """Rewrites the query if retrieval is not successful."""
        messages = state["messages"]
        question = messages[-1].content

        rewrite_prompt = f"Rephrase the following query for better search results:\n\n{question}"
        llm = ChatGroq(model="qwen-2.5-32b")
        rewritten_query = llm.invoke(rewrite_prompt)

        return {"messages": messages + [HumanMessage(content=rewritten_query.content)]}

    # Define Retrieval
    def retrieve(state):
        """Retrieves relevant documents from FAISS."""
        messages = state["messages"]
        question = messages[-1].content
        retrieved_docs = retriever.invoke(question)

        if not retrieved_docs:
            print("❌ No relevant documents found for query:", question)
            return {"messages": messages + [AIMessage(content="Sorry, I couldn't find relevant information in the documents.")]}
        
        doc_context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        return {"messages": messages + [AIMessage(content=f"Retrieved Docs: {doc_context}")]}

    # Define Answer Generation
    def generate(state):
        """Generates a response using retrieved document context."""
        messages = state["messages"]
        question = messages[-1].content

        retrieved_docs = retriever.invoke(question)
        doc_context = "\n\n".join([doc.page_content for doc in retrieved_docs]) if retrieved_docs else "No relevant documents found."

        chat_history = "\n".join([
            f"User: {msg.content}" if isinstance(msg, HumanMessage) else f"AI: {msg.content}"
            for msg in messages
        ])

        full_prompt = f"""
        Previous Conversation:
        {chat_history}

        Retrieved Document Context:
        {doc_context}

        New User Question: {question}
        """

        llm = ChatGroq(model="qwen-2.5-32b")
        response = llm.invoke(full_prompt)

        return {"messages": messages + [AIMessage(content=response)]}

    # Define Graph Workflow
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("rewrite", rewrite)
    workflow.add_node("generate", generate)

    # Define workflow edges (as per the given workflow image)
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", tools_condition, {"tools": "retrieve", END: END})
    workflow.add_edge("retrieve", "rewrite")
    workflow.add_edge("rewrite", "agent")
    workflow.add_edge("generate", END)

    # Compile workflow
    graph = workflow.compile()

    # Chat Interface
    st.subheader("💬 Chat with Your Documents")

    # Display conversation history
    for msg in st.session_state["messages"]:
        if isinstance(msg, HumanMessage):
            st.markdown(f"**🧑‍💻 You:** {msg.content}")
        elif isinstance(msg, AIMessage):
            st.markdown(f"**🤖 AI:** {msg.content}")

    # User input field
    user_query = st.text_input("Ask a question:")

    if user_query:
        st.session_state["messages"].append(HumanMessage(content=user_query))

        # Run the agentic RAG workflow
        response = graph.invoke({"messages": st.session_state["messages"]})

        # Extract only the latest AI response
        ai_response = response["messages"][-1]

        # Store AI response in session state
        st.session_state["messages"].append(AIMessage(content=ai_response.content))

        # Display the AI response
        st.markdown(f"**🤖 AI:** {ai_response.content}")
