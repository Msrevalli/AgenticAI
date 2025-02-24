import os
import streamlit as st
from typing_extensions import TypedDict
from typing import Annotated, List
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langchain.prompts import PromptTemplate
import speech_recognition as sr
from io import BytesIO
import tempfile
import time

# Page configuration
st.set_page_config(
    page_title="AI Interview Assistant",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        font-weight: 800;
        margin-bottom: 1rem;
        text-align: center;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #4B5563;
        font-style: italic;
        margin-bottom: 2rem;
        text-align: center;
    }
    .question-header {
        background-color: #EFF6FF;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 5px solid #1E40AF;
    }
    .question-text {
        font-size: 1.3rem;
        font-weight: 600;
        color: #1E3A8A;
    }
    .question-number {
        font-size: 0.9rem;
        color: #4B5563;
        margin-bottom: 0.5rem;
    }
    .feedback-container {
        background-color: #F8FAFC;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border: 1px solid #E2E8F0;
    }
    .score-display {
        font-size: 1.2rem;
        font-weight: 600;
        text-align: center;
        margin: 1rem 0;
    }
    .high-score {
        color: #059669;
    }
    .medium-score {
        color: #D97706;
    }
    .low-score {
        color: #DC2626;
    }
    .final-evaluation {
        background-color: #F0FDF4;
        padding: 2rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border: 1px solid #D1FAE5;
    }
    .btn-primary {
        background-color: #1E40AF;
        color: white;
        font-weight: 600;
        padding: 0.5rem 1rem;
        border-radius: 0.3rem;
        border: none;
        transition: background-color 0.3s;
    }
    .btn-primary:hover {
        background-color: #1E3A8A;
    }
    .section-divider {
        margin: 2rem 0;
        border-top: 1px solid #E5E7EB;
    }
    .stAudio {
        margin: 1rem 0;
    }
    .footer {
        text-align: center;
        margin-top: 2rem;
        color: #6B7280;
        font-size: 0.8rem;
    }
    .stTextArea textarea {
        border-radius: 0.5rem;
        border: 1px solid #D1D5DB;
    }
    /* Make the recording button more visible */
    .stAudioInput button {
        background-color: #EF4444 !important;
        color: white !important;
    }
    /* Custom card layout */
    .card {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: white;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Load API Keys (Set up in .env file)

os.environ["GROQ_API_KEY"] = st.secrets("GROQ_API_KEY")

# Initialize LLM
llm = ChatGroq(model="qwen-2.5-32b")

# Define AI State
class InterviewState(TypedDict):
    job_description: str
    interview_questions: List[str]
    current_question: str
    answer: str
    feedback: str
    score: int
    final_feedback: str
    current_question_index: int
    max_questions: int
    interview_complete: bool
    previous_answers: List[dict]

# Step 1: Generate Questions
generate_questions_prompt = PromptTemplate(
    input_variables=["job_description"],
    template="""Based on the following job description, generate 5 interview questions in a numbered format:\n\n"
        "{job_description}\n\n"
        "Format the output as:\n"
        "1. [Question 1]\n"
        "2. [Question 2]\n"
        "3. [Question 3]\n"
        "4. [Question 4]\n"
        "5. [Question 5]"
 """
)
generate_questions_chain = generate_questions_prompt | llm

def generate_questions(state: InterviewState):
    response = generate_questions_chain.invoke({"job_description": state["job_description"]})
    questions = response.content.strip().split('\n')
    return {
        "interview_questions": questions, 
        "current_question": questions[0], 
        "current_question_index": 0,
        "max_questions": len(questions),
        "interview_complete": False,
        "previous_answers": []
    }

# Step 2: Analyze Answer
analyze_answer_prompt = PromptTemplate(
    input_variables=["current_question", "answer"],
    template="Evaluate this answer based on clarity, correctness, and depth.\nQuestion: {current_question}\nAnswer: {answer}\nProvide a score out of 5 as a single number on a new line."
)
analyze_answer_chain = analyze_answer_prompt | llm

def analyze_answer(state: InterviewState):
    response = analyze_answer_chain.invoke({
        "current_question": state["current_question"], 
        "answer": state["answer"]
    })
    lines = response.content.strip().split('\n')
    score = int(lines[-1]) if lines[-1].isdigit() else 5
    state["score"] = score
    return state

# Step 3: Provide Feedback
feedback_prompt = PromptTemplate(
    input_variables=["answer", "score"],
    template="Provide constructive feedback on this answer based on its score ({score}/5).\nAnswer: {answer}"
)
feedback_chain = feedback_prompt | llm

def provide_feedback(state: InterviewState):
    response = feedback_chain.invoke({
        "answer": state["answer"], 
        "score": state["score"]
    })
    new_previous_answers = state.get("previous_answers", []).copy()
    new_previous_answers.append({
        "question": state["current_question"],
        "answer": state["answer"],
        "feedback": response.content,
        "score": state["score"]
    })
    state["feedback"] = response.content
    state["previous_answers"] = new_previous_answers
    return state

# Modified to always go next regardless of score
def route_after_feedback(state: InterviewState):
    if state["current_question_index"] >= state["max_questions"] - 1:
        return "finish"
    return "next"

def next_question(state: InterviewState):
    new_index = state["current_question_index"] + 1
    state["current_question_index"] = new_index
    if new_index < len(state["interview_questions"]):
        state["current_question"] = state["interview_questions"][new_index]
        state["answer"] = ""
    else:
        state["interview_complete"] = True
    return state

# Step 4: Final Feedback
final_feedback_prompt = PromptTemplate(
    input_variables=["previous_answers"],
    template="""Based on the interview performance, provide a final evaluation.
    
Previous answers and scores:
{previous_answers}

Give an overall assessment of the candidate's performance.
"""
)
final_feedback_chain = final_feedback_prompt | llm

def generate_final_feedback(state: InterviewState):
    previous_answers_text = ""
    total_score = 0
    for i, ans in enumerate(state["previous_answers"]):
        previous_answers_text += f"Question {i+1}: {ans['question']}\n"
        previous_answers_text += f"Answer: {ans['answer']}\n"
        previous_answers_text += f"Score: {ans['score']}/10\n\n"
        total_score += ans['score']
    avg_score = total_score / len(state["previous_answers"]) if state["previous_answers"] else 0
    previous_answers_text += f"Average score: {avg_score:.1f}/10"
    response = final_feedback_chain.invoke({"previous_answers": previous_answers_text})
    return {
        "final_feedback": response.content,
        "interview_complete": True
    }

# Speech Recognition
def recognize_speech_from_mic(audio_file):
    recognizer = sr.Recognizer()
    with tempfile.NamedTemporaryFile(delete=True, suffix='.wav') as temp_audio:
        temp_audio.write(audio_file.read())
        temp_audio.flush()
        temp_audio.seek(0)  # Reset file pointer to the beginning
        with sr.AudioFile(temp_audio) as source:
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio)
        return text
    except sr.RequestError:
        return "API unavailable"
    except sr.UnknownValueError:
        return "Unable to recognize speech"

# Helper function to get score CSS class
def get_score_class(score):
    if score >= 8:
        return "high-score"
    elif score >= 6:
        return "medium-score"
    else:
        return "low-score"

# Build Workflow
workflow = StateGraph(InterviewState)
workflow.add_node("generate_questions", generate_questions)
workflow.add_edge(START, "generate_questions")
workflow.add_edge("generate_questions", END)

answer_workflow = StateGraph(InterviewState)
answer_workflow.add_node("analyze_answer", analyze_answer)
answer_workflow.add_node("provide_feedback", provide_feedback)
answer_workflow.add_node("next_question", next_question)
answer_workflow.add_node("generate_final_feedback", generate_final_feedback)
answer_workflow.add_edge(START, "analyze_answer")
answer_workflow.add_edge("analyze_answer", "provide_feedback")
answer_workflow.add_conditional_edges(
    "provide_feedback",
    route_after_feedback,
    {"next": "next_question", "finish": "generate_final_feedback"}
)
answer_workflow.add_edge("next_question", END)
answer_workflow.add_edge("generate_final_feedback", END)

interview_graph = workflow.compile()
answer_graph = answer_workflow.compile()

# Initialize session state
if "interview_started" not in st.session_state:
    st.session_state.interview_started = False
    st.session_state.interview_state = None
    st.session_state.submitted_answer = False
    st.session_state.answer_text = ""
    st.session_state.recording = False
    st.session_state.audio_bytes = None

# App Header
st.markdown('<div class="main-header">🎙️ AI Interview Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Practice your interview skills with personalized AI feedback</div>', unsafe_allow_html=True)

# Sidebar with instructions
with st.sidebar:
    st.markdown("### How It Works")
    st.markdown("""
    1. **Enter Job Description** - Paste a real job posting or describe your target role
    2. **Start Interview** - AI will generate relevant interview questions
    3. **Record Your Answers** - Speak your responses naturally
    4. **Get Feedback** - Receive personalized evaluation and tips
    5. **Review Performance** - See your overall results at the end
    """)
    
    st.markdown("### Tips for Best Results")
    st.markdown("""
    - Use a quiet environment for better audio recognition
    - Speak clearly and at a normal pace
    - Structure your answers with an introduction, main points, and conclusion
    - Use specific examples from your experience
    """)
    
    st.markdown("### About")
    st.markdown("""
    This AI Interview Assistant uses natural language processing to evaluate your answers based on:
    - Relevance to the question
    - Structure and clarity
    - Technical accuracy
    - Depth of knowledge
    """)

# Main content area
if not st.session_state.interview_started:
    # Job description input
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Job Description")
    st.markdown("Enter a job description or skills you want to practice interviewing for:")
    job_description = st.text_area("", "Looking for a Python Developer with experience in Flask, SQL, and REST APIs.", height=150)
    
    # Start interview button
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        if st.button("🚀 Start Interview", type="primary"):
            initial_state = {
                "job_description": job_description,
                "interview_questions": [],
                "current_question": "",
                "answer": "",
                "feedback": "",
                "score": 0,
                "final_feedback": "",
                "current_question_index": 0,
                "max_questions": 0,
                "interview_complete": False,
                "previous_answers": []
            }
            with st.spinner("Generating interview questions..."):
                interview_state = interview_graph.invoke(initial_state)
            st.session_state.interview_started = True
            st.session_state.interview_state = interview_state
            st.session_state.submitted_answer = False
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    

else:
    state = st.session_state.interview_state
    
    if state.get("interview_complete", False):
        # Final evaluation section
        st.markdown('<div class="final-evaluation">', unsafe_allow_html=True)
        st.markdown("### 🏆 Interview Complete!")
        
        # Calculate average score
        total_score = sum(ans['score'] for ans in state.get("previous_answers", []))
        avg_score = total_score / len(state.get("previous_answers", [])) if state.get("previous_answers", []) else 0
        
        # Display average score with color coding
        score_class = get_score_class(avg_score)
        st.markdown(f'<div class="score-display {score_class}">Overall Score: {avg_score:.1f}/10</div>', unsafe_allow_html=True)
        
        # Final feedback
        st.markdown("### Final Evaluation")
        st.markdown(state.get("final_feedback", "Interview completed."))
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Interview summary with expandable sections
        st.markdown("### Interview Summary")
        
        for i, ans in enumerate(state.get("previous_answers", [])):
            score_class = get_score_class(ans['score'])
            with st.expander(f"Question {i+1}: {ans['question']}"):
                st.markdown(f"**Your Answer:**")
                st.markdown(f"{ans['answer']}")
                st.markdown(f'<div class="score-display {score_class}">Score: {ans["score"]}/5</div>', unsafe_allow_html=True)
                st.markdown(f"**Feedback:**")
                st.markdown(f"{ans['feedback']}")
        
        # Start new interview button
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            if st.button("🔄 Start New Interview", type="primary"):
                st.session_state.interview_started = False
                st.session_state.interview_state = None
                st.session_state.submitted_answer = False
                st.rerun()
    else:
        # Interview in progress
        curr_idx = state["current_question_index"]
        max_questions = len(state["interview_questions"])
        
        # Progress indicator
        progress = (curr_idx + 1) / max_questions
        st.progress(progress)
        
        # Question display
        st.markdown('<div class="question-header">', unsafe_allow_html=True)
        st.markdown(f'<div class="question-number">Question {curr_idx + 1} of {max_questions}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="question-text">{state["current_question"]}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        if st.session_state.submitted_answer:
            # Answer and feedback display
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### Your Answer:")
            st.markdown(st.session_state.answer_text)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Feedback with score
            st.markdown('<div class="feedback-container">', unsafe_allow_html=True)
            st.markdown("### Feedback:")
            st.markdown(state["feedback"])
            
            # Display score with appropriate color
            score_class = get_score_class(state["score"])
            st.markdown(f'<div class="score-display {score_class}">Score: {state["score"]}/10</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Continue button
            col1, col2, col3 = st.columns([1,2,1])
            with col2:
                button_text = "Continue to Next Question" if curr_idx < max_questions - 1 else "Complete Interview"
                if st.button(button_text, type="primary"):
                    if curr_idx < max_questions - 1:
                        # Use the answer_graph to properly update the state
                        with st.spinner("Preparing next question..."):
                            st.session_state.interview_state = state
                            st.session_state.submitted_answer = False
                    else:
                        # Complete interview - generate final feedback
                        with st.spinner("Generating final evaluation..."):
                            final_state = generate_final_feedback(state.copy())
                            updated_state = state.copy()
                            updated_state.update(final_state)
                            st.session_state.interview_state = updated_state
                    st.rerun()
        else:
            # Answer recording section
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### Speak Your Answer:")
            st.markdown("Click the microphone button below to record your response.")
            
            col1, col2 = st.columns([3, 1])
            with col1:
                audio_data = st.audio_input("Record")
            
            if audio_data:
                st.audio(audio_data, format="audio/wav")
                with st.spinner("Transcribing your answer..."):
                    transcribed_text = recognize_speech_from_mic(audio_data)
                
                st.markdown("### Transcribed Answer:")
                st.write(transcribed_text)
                
                col1, col2, col3 = st.columns([1,2,1])
                with col2:
                    if st.button("Submit Answer", type="primary"):
                        st.session_state.answer_text = transcribed_text
                        eval_state = state.copy()
                        eval_state["answer"] = transcribed_text
                        with st.spinner("Analyzing your answer..."):
                            updated_state = answer_graph.invoke(eval_state)
                        st.session_state.interview_state = updated_state
                        st.session_state.submitted_answer = True
                        st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown('<div class="footer">AI Interview Assistant © 2025 | Powered by LangChain and Groq</div>', unsafe_allow_html=True)