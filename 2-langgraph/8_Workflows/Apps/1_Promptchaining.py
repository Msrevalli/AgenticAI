import streamlit as st
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
from langchain_groq import ChatGroq
import os


# -------------------- Load Environment Variables --------------------

os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

# Initialize LLM
llm = ChatGroq(model="qwen-2.5-32b")

# -------------------- Define Graph State --------------------
class State(TypedDict):
    pitch_text: str
    key_insights: str
    clarity_report: str
    storytelling_suggestions: str
    persuasive_tweaks: str
    restructuring_advice: str
    investor_questions: str
    final_enhanced_pitch: str
    pitch_category: str
    pitch_feedback: str

# -------------------- Workflow Nodes --------------------
def extract_key_insights(state: State):
    text = state["pitch_text"]
    prompt = f"""
    Analyze the following business pitch and extract the 5 most important insights:
    - Business model and revenue streams
    - Target market and size
    - Competitive advantage and differentiation
    - Team qualifications and experience
    - Growth strategy and financial projections

    PITCH:
    {text}

    Format your response as bullet points for each category.
    """
    response = llm.invoke(prompt)
    state["key_insights"] = response.content
    return state

def check_clarity(state: State):
    insights = state["key_insights"]
    prompt = f"""
    Evaluate the following business pitch insights for clarity and structure on a scale of 1-10:

    INSIGHTS:
    {insights}

    Assess the following aspects:
    1. Value proposition clarity (1-10)
    2. Market opportunity explanation (1-10)
    3. Business model comprehensibility (1-10)
    4. Competitive analysis clarity (1-10)
    5. Financial projections realism (1-10)

    For each aspect, provide a specific improvement suggestion.
    """
    response = llm.invoke(prompt)
    state["clarity_report"] = response.content
    return state

def categorize_pitch(state: State):
    clarity = state["clarity_report"]
    prompt = f"""
    Evaluate this business pitch comprehensively and categorize it into exactly ONE of these categories:
    - **Pass**: Ready for investors with minimal changes. The pitch clearly communicates value proposition, market opportunity, business model, competitive advantage, and financial projections.
    - **Needs Improvement**: Shows potential but requires refinement in key areas. May have strong elements but lacks clarity or persuasiveness in others.
    - **Fail**: Requires major restructuring before being investor-ready. Multiple fundamental elements are missing or poorly articulated.

    CLARITY ANALYSIS:
    {clarity}

    First, provide your category determination as a single word: "Pass", "Needs Improvement", or "Fail".
    Then, provide 3-5 specific reasons for this categorization.
    """
    response = llm.invoke(prompt)
    if "Pass" in response.content:
        state["pitch_category"]="Pass"
    elif "Needs Improvement" in response.content:
        state["pitch_category"]="Needs Improvement"
    else :
        state["pitch_category"]="Fail"
      
    feedback = response.content
    
    state["pitch_feedback"] = feedback.strip()
    return state

def enhance_storytelling(state: State):
    clarity = state["clarity_report"]
    prompt = f"""
    Transform this business pitch into a compelling narrative by:
    1. Creating a strong opening hook about the problem being solved
    2. Developing a coherent story arc (problem → solution → impact)
    3. Adding 1-2 concrete customer examples or use cases
    4. Incorporating emotional elements that resonate with investors
    5. Ensuring a strong closing that inspires action

    CURRENT PITCH:
    {clarity}

Rewrite the pitch maintaining all business details while enhancing the storytelling.
"""
    response = llm.invoke(prompt)
    state["storytelling_suggestions"] = response.content
    return state

def restructure_pitch(state: State):
    clarity = state["clarity_report"]
    prompt = f"""
        This business pitch needs significant restructuring. Create a detailed improvement plan addressing these elements:

        1. Executive Summary: How to concisely state the value proposition in 1-2 sentences
        2. Problem Statement: How to clearly articulate the market pain point
        3. Solution: How to present the product/service more effectively
        4. Market Analysis: How to demonstrate market size and growth potential
        5. Business Model: How to clarify revenue streams and unit economics
        6. Competitive Landscape: How to position against alternatives
        7. Team: How to highlight relevant expertise and experience
        8. Financials: How to present realistic projections
        9. Ask: How to clearly state what is being requested from investors

        For each element, provide:
        - Current issue in the pitch
        - Specific restructuring recommendation
        - Example language/phrasing they could use

        PITCH ANALYSIS:
        {clarity}
        """
    response = llm.invoke(prompt)
    state["restructuring_advice"] = response.content
    return state

def refine_persuasiveness(state: State):
    content1 = state.get("storytelling_suggestions", "")
    content2 = state.get("restructuring_advice", "")
    prompt = f"""
    Enhance the persuasiveness of this business pitch by incorporating:

    1. Data points and statistics that validate the market opportunity
    2. Social proof elements (potential customer testimonials, expert endorsements)
    3. Scarcity or urgency factors that motivate investment
    4. Clear ROI projections or exit strategy possibilities
    5. Strong calls to action for investors

    CURRENT PITCH:
    {content1}
    {content2}

    Rewrite the pitch maintaining its structure while adding these persuasive elements. Use specific numbers and metrics where possible, and ensure all claims are backed by evidence or reasoning.
    """
    response = llm.invoke(prompt)
    state["persuasive_tweaks"] = response.content
    return state

def generate_investor_questions(state: State):
    if state["persuasive_tweaks"]:
        persuasive_pitch = state["persuasive_tweaks"]

    else:
        persuasive_pitch=state['pitch_text']
    prompt = f"""
    Based on this business pitch, generate 6-8 challenging questions that sophisticated investors would likely ask, focusing on:

    1. Market validation and customer acquisition
    2. Financial projections and unit economics
    3. Competitive threats and barriers to entry
    4. Team capabilities and experience gaps
    5. Regulatory or scaling challenges
    6. Exit strategy and return potential

    For each question:
    1. Phrase it as an investor would actually ask it
    2. Provide a strong, data-backed answer (1-2 paragraphs)
    3. Include specific metrics, comparisons, or examples that strengthen credibility

    PITCH:
    {persuasive_pitch}

    Format as "Q: [Question]" followed by "A: [Answer]" for each.
    """
    response = llm.invoke(prompt)
    state["investor_questions"] = response.content
    return state

def finalize_pitch(state: State):
    persuasive = state["persuasive_tweaks"]
    q_and_a = state["investor_questions"]
    prompt = f"""
    Create a comprehensive, investor-ready pitch deck outline incorporating all improvements. Structure it as follows:

    1. Hook/Opening (10-15 words) - Attention-grabbing statement
    2. Problem (2-3 sentences) - Clear market pain point
    3. Solution (3-4 sentences) - Your product/service explained simply
    4. Market Size (2-3 sentences with specific TAM/SAM/SOM figures)
    5. Business Model (2-3 sentences with clear revenue streams)
    6. Traction (2-3 key metrics or milestones achieved)
    7. Competitive Advantage (2-3 sentences on differentiation)
    8. Team (1-2 sentences highlighting key expertise)
    9. Financials (3-4 key projections for next 3 years)
    10. Ask (1-2 sentences on investment needed and use of funds)
    11. Vision/Close (1 powerful sentence on future impact)

    Include a "Handling Objections" section addressing the top 3 investor concerns.

    PERSUASIVE ELEMENTS:
    {persuasive}

    INVESTOR Q&A:
    {q_and_a}

    Format this as a complete pitch with clear section headers and concise, impactful content under each.
    """
    response = llm.invoke(prompt)
    state["final_enhanced_pitch"] = response.content
    return state

# -------------------- Build LangGraph Workflow --------------------
graph = StateGraph(State)

graph.add_node("extract_key_insights", extract_key_insights)
graph.add_node("check_clarity", check_clarity)
graph.add_node("categorize_pitch", categorize_pitch)
graph.add_node("enhance_storytelling", enhance_storytelling)
graph.add_node("restructure_pitch", restructure_pitch)
graph.add_node("refine_persuasiveness", refine_persuasiveness)
graph.add_node("generate_investor_questions", generate_investor_questions)
graph.add_node("finalize_pitch", finalize_pitch)

graph.add_edge(START, "extract_key_insights")
graph.add_edge("extract_key_insights", "check_clarity")
graph.add_edge("check_clarity", "categorize_pitch")

def category_based_routing(state: State):
    category = state["pitch_category"].strip().lower()
    if "pass" in category:
        return "Pass"
    elif "needs improvement" in category:
        return "Needs Improvement"
    else:
        return "Fail"

graph.add_conditional_edges("categorize_pitch", category_based_routing, {
    "Pass": "generate_investor_questions",
    "Needs Improvement": "enhance_storytelling",
    "Fail": "restructure_pitch",
})

graph.add_edge("generate_investor_questions", "finalize_pitch")
graph.add_edge("enhance_storytelling", "refine_persuasiveness")
graph.add_edge("restructure_pitch", "refine_persuasiveness")
graph.add_edge("refine_persuasiveness", "generate_investor_questions")
graph.add_edge("finalize_pitch", END)

workflow = graph.compile()

import pypdf


def extract_text_from_file(file):
    if file.type == "application/pdf":
        pdf_reader = pypdf.PdfReader(file)
        return " ".join([page.extract_text() for page in pdf_reader.pages])
    
    


# -------------------- Streamlit UI --------------------
st.title("🚀 AI-Powered Business Pitch Evaluator")
st.write("This app analyzes your business pitch and provides insights, clarity checks, and investor-focused improvements.")

uploaded_file = st.file_uploader("Upload Business Pitch (PDF)", type=["pdf"])

if uploaded_file is not None:
    pitch_text = extract_text_from_file(uploaded_file)
else:
    pitch_text = st.text_area("Or enter your pitch manually:")

if st.button("Analyze Pitch"):
    with st.spinner("Processing your pitch..."):
        state = {"pitch_text": pitch_text}
        result = workflow.invoke(state)
    
        # Display results
        st.subheader("📌 **Pitch Analysis Results**")
        st.write(f"🔍 **Pitch Category:** {result['pitch_category']}")
        st.write(f"📝 **Feedback:** {result['pitch_feedback']}")
        
        with st.expander("📌 Key Insights"):
            st.write(result["key_insights"])
        
        with st.expander("📌 Clarity Report"):
            st.write(result["clarity_report"])
        
        with st.expander("📌 Storytelling Improvements"):
            st.write(result.get("storytelling_suggestions", "No storytelling enhancements needed."))
        
        with st.expander("📌 Persuasion Refinements"):
            st.write(result.get("persuasive_tweaks", "No persuasion refinements needed."))
        
        with st.expander("📌 Investor Q&A"):
            st.write(result["investor_questions"])
        
        with st.expander("📌 Final Enhanced Pitch"):
            st.write(result["final_enhanced_pitch"])

        st.download_button("📥 Download Enhanced Pitch", result["final_enhanced_pitch"], file_name="enhanced_pitch.txt")
        st.markdown("### 🔗 Powered by LangGraph with Prompt Chaining Workflow 🚀")
        st.write("This AI-driven app analyzes and improves business pitches using advanced prompt chaining techniques.")


# ✅ Display Workflow Diagram in Sidebar
with st.sidebar:
    st.subheader("Workflow Diagram")

    # ✅ Generate Mermaid Workflow Diagram
    mermaid_diagram = workflow.get_graph().draw_mermaid_png()

    # ✅ Save and Display the Image in Sidebar
    image_path = "workflow_diagram.png"
    with open(image_path, "wb") as f:
        f.write(mermaid_diagram)

    st.image(image_path, caption="Workflow Execution")

    




