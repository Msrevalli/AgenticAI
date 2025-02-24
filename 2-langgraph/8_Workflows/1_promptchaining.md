# **🚀 Understanding Prompt Chaining in AI Workflows**
### **📌 What is Prompt Chaining?**
**Prompt chaining** is a technique where **multiple prompts** are **linked together in a structured sequence** to refine, enhance, or guide AI-generated responses.

Instead of providing a **single, complex prompt**, prompt chaining **breaks down a task into smaller steps**, where **each step builds upon the previous one**.

### **🔹 Benefits of Prompt Chaining**
✅ **Better Accuracy** – AI **refines responses iteratively**, reducing hallucinations.  
✅ **Task Breakdown** – Complex problems **become manageable**, improving AI output.  
✅ **Controlled AI Responses** – Each step **guides AI** to stay on topic.  
✅ **Memory Handling** – AI **remembers previous context** for multi-turn interactions.  

---

## **📌 How Prompt Chaining Works in LangGraph?**
LangGraph uses **graph-based execution** where **nodes (functions) are connected as a sequence**.  
Unlike **linear LangChain chains**, LangGraph allows:
- **More flexible chaining** (e.g., loops, conditional branching).
- **Parallel execution of prompts** (instead of just sequential processing).
- **State tracking** for **memory & multi-step workflows**.

### **🔹 Steps to Use Prompt Chaining in LangGraph**
1. **Define State** → Store and track **conversation context**.  
2. **Create Nodes** → Each **prompt step** runs as a **separate function**.  
3. **Define Edges** → Link **nodes together** to form a **chain-like workflow**.  
4. **Compile & Execute** → The graph executes **each step sequentially**.

---

## **🚀 Example: Prompt Chaining for AI Content Refinement**
We will build a **3-step LangGraph-based prompt chain**:
1️⃣ **Extract Keywords** – Identify main concepts from user input.  
2️⃣ **Generate Summary** – Use extracted keywords to create a short summary.  
3️⃣ **Generate AI-Enhanced Response** – Convert the summary into a **detailed response**.

### **🔹 Execution Flow:**
```
START → Extract Keywords → Generate Summary → AI-Enhanced Response → END
```

---

### **📌 Full Code for Prompt Chaining in LangGraph**
```python
# 📌 Import Required Libraries
import os
from dotenv import load_dotenv
from typing_extensions import TypedDict
from typing import Annotated
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# ✅ Load Environment Variables (for OpenAI API Key)
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# ✅ Initialize LLM (GPT-4)
llm = ChatOpenAI(model="gpt-4", temperature=0)

# ✅ Define AI State (Memory for Workflow)
class AIState(TypedDict):
    messages: Annotated[list, add_messages]

# ✅ Step 1: Extract Keywords Prompt
extract_keywords_prompt = PromptTemplate(
    input_variables=["text"],
    template="Extract the main keywords from the following text:\n{text}"
)
extract_chain = LLMChain(llm=llm, prompt=extract_keywords_prompt)

def extract_keywords(state: AIState):
    """Extracts keywords from user input"""
    keywords = extract_chain.run(state["messages"][-1])  # Last message contains user input
    return {"messages": state["messages"] + [keywords]}  # Adds keywords to state

# ✅ Step 2: Generate Summary Prompt
summary_prompt = PromptTemplate(
    input_variables=["keywords"],
    template="Using these keywords: {keywords}, generate a concise summary."
)
summary_chain = LLMChain(llm=llm, prompt=summary_prompt)

def generate_summary(state: AIState):
    """Generates a summary using extracted keywords"""
    summary = summary_chain.run(state["messages"][-1])  # Uses extracted keywords
    return {"messages": state["messages"] + [summary]}  # Adds summary to state

# ✅ Step 3: Generate AI-Enhanced Response
final_prompt = PromptTemplate(
    input_variables=["summary"],
    template="Use this summary: {summary}, and generate a detailed AI-enhanced response."
)
final_chain = LLMChain(llm=llm, prompt=final_prompt)

def enhanced_response(state: AIState):
    """Enhances the summary into a full AI-generated response"""
    response = final_chain.run(state["messages"][-1])  # Uses summary
    return {"messages": state["messages"] + [response]}  # Adds final response

# ✅ Build LangGraph Workflow
workflow = StateGraph(AIState)

# ➤ Add Nodes
workflow.add_node("extract_keywords", extract_keywords)
workflow.add_node("generate_summary", generate_summary)
workflow.add_node("enhanced_response", enhanced_response)

# ➤ Define Execution Flow
workflow.add_edge(START, "extract_keywords")  # Start → Extract Keywords
workflow.add_edge("extract_keywords", "generate_summary")  # Keywords → Summary
workflow.add_edge("generate_summary", "enhanced_response")  # Summary → Final Response
workflow.add_edge("enhanced_response", END)  # Final Response → End

# ✅ Compile the Graph
graph = workflow.compile()

# ✅ Function to Run the Workflow
def process_input(user_input: str):
    """
    Runs the AI workflow with prompt chaining for a given input.
    """
    state = {"messages": [user_input]}  # Initialize chat state
    response = graph.invoke(state)

    for message in response["messages"]:
        print("🚀 AI Response:", message)

# ✅ Run Prompt Chaining AI Chatbot
if __name__ == "__main__":
    print("\n🤖 AI Chatbot with Prompt Chaining (LangGraph) | Type 'exit' to quit\n")

    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye! 👋")
                break

            process_input(user_input)

        except Exception as e:
            print("Error:", str(e))
            break
```

---

## **📌 How This LangGraph Workflow Works**
| **Step** | **Action** | **Example Output** |
|---------|-----------|------------------|
| **1️⃣ Extract Keywords** | AI extracts **key terms** from input | `"LangChain, AI, workflow"` |
| **2️⃣ Generate Summary** | AI creates a **short summary** using keywords | `"LangChain helps build AI workflows."` |
| **3️⃣ Enhance Response** | AI **expands** the summary into a detailed response | `"LangChain is a framework for structuring AI workflows using LLMs."` |

✔ **Each step refines the response for better accuracy!** 🚀  

---

## **📌 Expected Output**
```plaintext
🤖 AI Chatbot with Prompt Chaining (LangGraph) | Type 'exit' to quit

User: What is LangChain?
🚀 AI Response: LangChain, AI, LLM, framework
🚀 AI Response: LangChain is an AI framework for building LLM applications.
🚀 AI Response: LangChain simplifies AI integration by providing structured workflows for LLM-powered applications.
```
✔ **AI automatically sequences prompt execution!** 🎉  

---

## **📌 When to Use Prompt Chaining in LangGraph?**
| **Use Case** | **Why Use It?** |
|-------------|------------------|
| **Complex AI Workflows** | Break down tasks into **smaller, manageable steps**. |
| **Multi-Step Reasoning** | AI **refines, verifies, and enhances** responses. |
| **Text Summarization & Analysis** | Extract key insights before **generating answers**. |
| **Legal & Finance Reports** | Ensure AI **processes facts before making conclusions**. |
| **Customer Support Chatbots** | Guide AI to **ask clarifying questions before responding**. |

---

# **🚀 Final Thoughts**
🔹 **Prompt chaining in LangGraph** enables **structured, iterative AI processing**.  
🔹 **Unlike LangChain chains**, LangGraph allows **conditional logic, loops, and multi-agent workflows**.  
🔹 **Useful for summarization, report generation, and advanced chatbot logic.**

