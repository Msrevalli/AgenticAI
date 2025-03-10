import os
from dotenv import load_dotenv
from typing_extensions import TypedDict
from typing import List, Dict
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph, START, END
from langchain_core.output_parsers import StrOutputParser

# Load API keys
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Initialize LLM
llm = ChatGroq(model="gemma2-9b-it")  # Ensure model name is correct

# Define State
class State(TypedDict):
    user_requirements: str
    user_stories: List[str]
    PO_review_feedback: str  # Feedback from the Product Owner
    PO_approved: bool  # If True, end process
    design_docs: Dict[str, Dict[str, str]]  # Dictionary with functional & technical docs
    design_review_feedback:str
    design_review_approval:bool

# Function to Generate User Stories
def create_stories(state: State):
    """Generates user stories from user requirements using LLM."""
    userstories_prompt = PromptTemplate(
        template="""You are an expert software developer. Based on the following requirement:
        "{user_requirements}", generate exactly 5 well-structured user stories.
        
        Format:
        1. As a <user>, I want to <action> so that <benefit>.

        Provide only the user stories, nothing else.""",
        input_variables=["user_requirements"]
    )

    userstory_chain = userstories_prompt | llm | StrOutputParser()
    response = userstory_chain.invoke({"user_requirements": state["user_requirements"]})
    user_stories = response.strip().split("\n") 

    return {"user_stories": user_stories}

# Function for Product Owner Review
def product_owner_review(state: State):
    """Product Owner reviews user stories and provides feedback."""
    print("\n📌 Product Owner Review: Please review the following user stories.\n")
    for story in state["user_stories"]:
        print(f"{story}")

    status = input("Status (Approved/needing improvement): ").strip().lower()

    if status == "approved":
        return {"PO_review_feedback": "Approved", "PO_approved": True}
    else:
        feedback = input("Provide review remarks: ").strip()
        return {"PO_review_feedback": feedback, "PO_approved": False}

# Function to Regenerate User Stories Based on Product Owner Feedback
def regenerate_stories(state: State):
    """Regenerates user stories based on PO feedback."""
    feedback_summary = state["PO_review_feedback"]
    old_stories_context = "\n".join(state["user_stories"])

    regeneration_prompt = PromptTemplate(
        template="""You are an expert in generating user stories. The Product Owner has provided feedback.
        
        Below is the feedback:
        {feedback_summary}
        
        These are the previously generated user stories:
        {old_stories_context}
        
        Based on this feedback, regenerate exactly 5 user stories that incorporate these improvements.
        
        Format:
        1. As a <user>, I want to <action> so that <benefit>.
        
        Only provide the updated user stories, nothing else.""",
        input_variables=["feedback_summary", "old_stories_context"]
    )

    regeneration_chain = regeneration_prompt | llm | StrOutputParser()
    response = regeneration_chain.invoke({
        "feedback_summary": feedback_summary,
        "old_stories_context": old_stories_context
    })

    regenerated_stories = response.strip().split("\n") if response else []
    print("Regenerated Stories:")

    print(f'{regenerate_stories}\n')
    return {"user_stories": regenerated_stories}

# Function to Create Design Documents (Functional & Technical)
def create_design_documents(state: State):
    """Generates functional & technical design docs based on user stories."""

    # Functional Design Prompt
    functional_prompt = PromptTemplate(
        template="""You are an expert in generating functional design documents based on user stories.
        
        Provide a detailed functional design document for the following user stories:
        {user_stories}

        Format:
        - Purpose
        - Features
        - User Flow
        - Expected Outcomes""",
        input_variables=["user_stories"]
    )

    # Technical Design Prompt
    technical_prompt = PromptTemplate(
        template="""You are an expert in generating technical design documents based on user stories.
        
        Provide a detailed technical design document for the following user stories:
        {user_stories}

        Format:
        - Technology Stack
        - System Architecture
        - Database Design
        - APIs and Integrations""",
        input_variables=["user_stories"]
    )

    # Generate Functional Design
    functional_chain = functional_prompt | llm | StrOutputParser()
    functional_response = functional_chain.invoke({"user_stories": "\n".join(state["user_stories"])})

    # Generate Technical Design
    technical_chain = technical_prompt | llm | StrOutputParser()
    technical_response = technical_chain.invoke({"user_stories": "\n".join(state["user_stories"])})
    print('Design Docs are\n')
    print(functional_response)
    print('\n')
    print(technical_response)

    return {
        "design_docs": {
            "functional": functional_response,
            "technical": technical_response
        }
    }

# Function for Technical Reviewer to Review Design
def technical_reviewer(state: State):
    """Technical Reviewer reviews the design documents and provides feedback."""
    print("\n📌 Technical Review: Reviewing Functional & Technical Design Documents\n")

    print("\n📄 Functional Design:")
    print(state["design_docs"]["functional"])
    print("\n⚙️ Technical Design:")
    print(state["design_docs"]["technical"])

    status = input("Status (Approved/Needs Improvement): ").strip().lower()

    if status == "approved":
        return {"design_review_feedback": "Approved", "design_review_approval": True}
    else:
        feedback = input("Provide review remarks: ").strip()
        return {"design_review_feedback": feedback, "design_review_approval": False}

# Function to Regenerate Design Docs Based on Reviewer Feedback
def regenerate_design_docs(state: State):
    """Regenerates Functional & Technical Design Docs based on Technical Review Feedback."""
    
    feedback_summary = state["design_review_feedback"]
    old_functional_design = state["design_docs"]["functional"]
    old_technical_design = state["design_docs"]["technical"]

    # **Functional Regeneration Prompt**
    functional_regen_prompt = PromptTemplate(
        template="""You are an expert in functional design. The Technical Reviewer has provided feedback.

        Below is the feedback:
        {feedback_summary}

        Previous Functional Design:
        {old_functional_design}

        Based on this feedback, regenerate an improved **functional** design document.

        Format:
        - **Purpose**: 
        - **Features**: 
        - **User Flow**: 
        - **Expected Outcomes**: """,
        input_variables=["feedback_summary", "old_functional_design"]
    )

    # **Technical Regeneration Prompt**
    technical_regen_prompt = PromptTemplate(
        template="""You are an expert in technical design. The Technical Reviewer has provided feedback.

        Below is the feedback:
        {feedback_summary}

        Previous Technical Design:
        {old_technical_design}

        Based on this feedback, regenerate an improved **technical** design document.

        Format:
        - **Technology Stack**: 
        - **System Architecture**: 
        - **Database Design**: 
        - **APIs and Integrations**: """,
        input_variables=["feedback_summary", "old_technical_design"]
    )

    # **Regenerate Functional & Technical Docs Separately**
    functional_chain = functional_regen_prompt | llm | StrOutputParser()
    technical_chain = technical_regen_prompt | llm | StrOutputParser()

    functional_response = functional_chain.invoke({
        "feedback_summary": feedback_summary,
        "old_functional_design": old_functional_design
    })

    technical_response = technical_chain.invoke({
        "feedback_summary": feedback_summary,
        "old_technical_design": old_technical_design
    })
    print(functional_response)
    print(technical_response)

    return {
        "design_docs": {
            "functional": functional_response.strip(),
            "technical": technical_response.strip()
        }
    }



# Create User Story Workflow
userstory_builder = StateGraph(State)
userstory_builder.add_node("create_stories", create_stories)
userstory_builder.add_node("product_owner_review", product_owner_review)
userstory_builder.add_node("regenerate_stories", regenerate_stories)

userstory_builder.add_edge(START, "create_stories")
userstory_builder.add_edge("create_stories", "product_owner_review")

userstory_builder.add_conditional_edges(
    "product_owner_review",
    lambda state: END if state["PO_approved"] else "regenerate_stories"
)

userstory_builder.add_edge("regenerate_stories", "product_owner_review")

# Compile User Story Graph
userstory_graph = userstory_builder.compile()

# Example Requirement
requirements = "Generate a snake game"

# Run User Story Graph (Including Product Owner Review)
output = userstory_graph.invoke({
    "user_requirements": requirements,
    "user_stories": [],
    "PO_review_feedback": "",
    "PO_approved": False,
    "design_docs": {},
    "design_review_feedback":"",
    "design_review_approval":False
})

# Create Design Workflow
design_builder = StateGraph(State)
design_builder.add_node("design_functiona_technical_docs", create_design_documents)
design_builder.add_node("Design_Review",technical_reviewer)
design_builder.add_node("regenerated_design_docs",regenerate_design_docs)
design_builder.add_edge(START, "design_functiona_technical_docs")
design_builder.add_edge("design_functiona_technical_docs","Design_Review")
design_builder.add_conditional_edges(
    "Design_Review",
    lambda state: END if state["design_review_approval"] else "regenerated_design_docs"
)


# Compile Design Graph
design_graph = design_builder.compile()

# Run Design Graph with Approved User Stories
design_output = design_graph.invoke({
    "user_requirements": requirements,
    "user_stories": output["user_stories"],
    "PO_review_feedback": output["PO_review_feedback"],
    "PO_approved": output["PO_approved"],
    "design_docs": {},
    "design_review_feedback":"",
    "design_review_approval":False
    
})


