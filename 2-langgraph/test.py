import os
import streamlit as st
from typing_extensions import TypedDict
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

# ✅ Load environment variables
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# ✅ Initialize LLM
llm = ChatGroq(model="qwen-2.5-32b")

# ✅ Define State for LangGraph
class State(TypedDict):
    messages: list  # Holds AI & User messages
    tool_call_approved: bool  # Indicates if human approved tool execution

# ✅ Define AI Assistant Node
def assistant(state: State):
    """AI generates a response, possibly requiring a tool call."""
    response = llm.invoke(state["messages"][-1])  # Pass latest user message
    return {"messages": state["messages"] + [response]}  # Append to conversation history

# ✅ Define Tool Execution Node
def tool_execution(state: State):
    """Executes tool action after human approval."""
    st.success("✅ Tool Execution in Progress...")
    return state  # Placeholder for real tool execution logic

# ✅ Define Human Review Node (Interrupt Before Execution)
def human_review(state: State):
    """Pause execution before tool call, allowing human review."""
    st.subheader("🛑 Human Review Required")
    st.write("AI suggested a tool call for:")
    st.write(state["messages"][-1])  # Display AI's last response

    if st.button("✅ Approve Tool Call"):
        return {"tool_call_approved": True}
    elif st.button("❌ Reject & Regenerate"):
        return {"tool_call_approved": False}

# ✅ Define Conditional Routing Logic
def should_continue(state: State):
    """Routes flow based on human approval."""
    return "tools" if state["tool_call_approved"] else "assistant"

# ✅ Build LangGraph Workflow
builder = StateGraph(State)

# ✅ Add Nodes
builder.add_node("assistant", assistant)
builder.add_node("human_review", human_review)
builder.add_node("tools", ToolNode([tool_execution]))

# ✅ Define Workflow Execution Order
builder.add_edge(START, "assistant")
builder.add_edge("assistant", "human_review")
builder.add_conditional_edges("human_review", should_continue, {"tools": "tools", "assistant": "assistant"})

# ✅ Compile Graph with Interrupt Handling
memory = MemorySaver()
graph = builder.compile(interrupt_before=["human_review"], checkpointer=memory)

# ✅ Streamlit UI Layout
st.set_page_config(page_title="Human-in-the-Loop AI", layout="wide")

st.title("🤖 AI Workflow with Human Approval for Tool Calls")
st.write("This app allows AI to suggest tool calls, but **pauses for human approval** before execution.")

# ✅ User Input for AI
user_input = st.text_input("Enter your query:", "")

if st.button("🚀 Start Workflow"):
    thread = {"configurable": {"thread_id": "1"}}

    # ✅ Run LangGraph workflow and pause for human review
    for event in graph.stream({"messages": [user_input]}, thread, stream_mode="values"):
        st.write(event)  # ✅ Display AI response

        # ✅ If interrupted for human review, wait for user input
        if "human_review" in event:
            st.warning("🚨 Workflow paused! Waiting for human approval...")
            break  # Stop execution until review is provided

# ✅ If workflow was interrupted, show review form
if st.session_state.get("tool_call_approved") is False:
    st.subheader("📝 Human Feedback Needed")
    
    if st.button("🔄 Resubmit"):
        graph.update_state(thread, {"tool_call_approved": True}, as_node="human_review")
        st.success("Tool execution approved! Restarting workflow...")
        st.rerun()
