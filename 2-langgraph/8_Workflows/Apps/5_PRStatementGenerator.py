from typing_extensions import TypedDict, Literal
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
import streamlit as st

import os
from dotenv import load_dotenv
load_dotenv()

os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
llm = ChatGroq(model="qwen-2.5-32b")

class State(TypedDict):
    pr_statement: str
    topic: str
    feedback: str
    quality: str

class Feedback(BaseModel):
    grade: Literal["good", "needs improvement"] = Field(
        description="Decide if the PR statement is well-formed or needs improvement.",
    )
    feedback: str = Field(
        description="If the PR statement needs improvement, provide feedback on how to refine it.",
    )

# Augment the LLM with schema for structured output
evaluator = llm.with_structured_output(Feedback)


def llm_call_generator(state: State):
    """LLM generates a PR statement"""

    if state.get("feedback"):
        msg = llm.invoke(
            f"Write a PR statement about {state['topic']} while considering this feedback: {state['feedback']}"
        )
    else:
        msg = llm.invoke(f"Write a PR statement about {state['topic']}")
    return {"pr_statement": msg.content}


def llm_call_evaluator(state: State):
    """LLM evaluates the PR statement"""

    grade = evaluator.invoke(f"Evaluate this PR statement: {state['pr_statement']}")
    return {"quality": grade.grade, "feedback": grade.feedback}


def route_pr_statement(state: State):
    """Route back to generator if PR needs refinement, or end if it is good"""

    if state["quality"] == "good":
        return "Accepted"
    elif state["quality"] == "needs improvement":
        return "Rejected + Feedback"


# Build workflow
optimizer_builder = StateGraph(State)

# Add the nodes
optimizer_builder.add_node("llm_call_generator", llm_call_generator)
optimizer_builder.add_node("llm_call_evaluator", llm_call_evaluator)

# Add edges to connect nodes
optimizer_builder.add_edge(START, "llm_call_generator")
optimizer_builder.add_edge("llm_call_generator", "llm_call_evaluator")
optimizer_builder.add_conditional_edges(
    "llm_call_evaluator",
    route_pr_statement,
    {
        "Accepted": END,
        "Rejected + Feedback": "llm_call_generator",
    },
)

# Compile the workflow
optimizer_workflow = optimizer_builder.compile()

st.title("📝 AI-Powered PR Statement Generator")

# Add sidebar with workflow diagram
with st.sidebar:
    st.subheader("Workflow Diagram")
    
    try:
        # Generate Mermaid Workflow Diagram
        mermaid_diagram = optimizer_workflow.get_graph().draw_mermaid_png()
        
        # Save and Display the Image in Sidebar
        image_path = "workflow_diagram.png"
        with open(image_path, "wb") as f:
            f.write(mermaid_diagram)
        
        st.image(image_path, caption="PR Statement Optimizer Workflow")
    except Exception as e:
        st.error(f"Unable to generate workflow diagram: {e}")
        st.info("The workflow still functions correctly even without visualization.")

# Main content area
st.markdown("This tool automatically generates and refines PR statements until they meet quality standards.")
topic = st.text_input("Enter the topic for the PR statement", "Company's AI-Powered Chatbot Launch")

# Track generation process
if st.button("Generate PR Statement"):
    with st.spinner("Generating optimized PR statement..."):
        # Create columns for showing generation progress
        col1, col2 = st.columns(2)
        
        # Initialize state
        progress_placeholder = st.empty()
        progress_placeholder.info("Starting PR statement generation...")
        
        # Invoke the workflow
        state = optimizer_workflow.invoke({"topic": topic})
        
        # Show success message
        progress_placeholder.success("PR statement successfully generated!")
        
        # Display the result
        st.subheader("Generated PR Statement:")
        st.write(state["pr_statement"])
        
        # Show feedback if available
        if state.get("feedback"):
            st.subheader("Improvement Feedback:")
            st.info(state["feedback"])