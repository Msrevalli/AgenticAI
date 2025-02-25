
## **1. Overview of the Notebook**
This notebook demonstrates how to use **LangGraph** to build an AI system that can:
- **Perform arithmetic operations** using LLM-integrated tools.
- **Interrupt execution** before responding, allowing human intervention (HITL).
- **Enable debugging & editing** of the AI’s decisions.
- **Visualize execution flow** of AI decision-making.

---
## **2. Key Components in the Notebook**
### **🔹 Loading Environment Variables**
The notebook first loads API keys:
```python
from dotenv import load_dotenv
load_dotenv()
import os
```
It sets up API keys for **Groq-based LLM (Qwen-2.5-32B)**:
```python
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
```
This ensures that the model can access cloud-based AI services.

---

### **🔹 Initializing the LLM (Groq Chat Model)**
The notebook uses the **Qwen-2.5-32B model** from Groq:
```python
from langchain_groq import ChatGroq

llm = ChatGroq(model="qwen-2.5-32b")
result = llm.invoke("Hello")
```
This tests the model’s ability to generate responses.

---

### **🔹 Defining Arithmetic Tools (Functions)**
Three arithmetic functions are defined to be used by the AI:
```python
def multiply(a: int, b: int) -> int:
    return a * b

def add(a: int, b: int) -> int:
    return a + b

def divide(a: int, b: int) -> float:
    return a / b
```
These functions are **tools that the LLM can call** when solving problems.

The LLM is then **bound to these tools**:
```python
tools = [add, multiply, divide]
llm_with_tools = llm.bind_tools(tools)
```
This means the AI can now **use these functions** when performing calculations.

---

### **🔹 Building the AI Workflow with LangGraph**
The notebook uses **LangGraph** to manage AI decision-making and human intervention.

**Importing LangGraph components:**
```python
from langgraph.graph import MessagesState, START, StateGraph
from langgraph.prebuilt import tools_condition, ToolNode
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
```
- `MessagesState`: Maintains the state of the conversation.
- `StateGraph`: Defines the execution flow.
- `ToolNode`: Enables AI to invoke predefined tools.
- `tools_condition`: Routes control based on whether the AI needs a tool.

**Defining a system message:**
```python
sys_msg = SystemMessage(content="You are a helpful assistant tasked with performing arithmetic on a set of inputs.")
```
This instructs the AI **how to behave** in the workflow.

---

### **🔹 Creating the Assistant Node**
The AI **processes user inputs** and generates responses:
```python
def assistant(state: MessagesState):
   return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}
```
- The **LLM is invoked with past messages** to maintain conversation context.
- The AI **calls the necessary arithmetic tool** when needed.

---

### **🔹 Defining the Workflow Graph**
A **StateGraph** is created to control execution:
```python
builder = StateGraph(MessagesState)
```
#### **Adding Nodes**
```python
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))
```
- **Assistant Node:** Calls the AI to process user inputs.
- **Tools Node:** Allows the AI to use arithmetic tools.

#### **Adding Edges (Execution Flow)**
```python
builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition, ["tools"])
builder.add_edge("tools", "assistant")
```
- The AI **starts execution at "assistant"**.
- If a **tool is required**, it moves to the "tools" node.
- The process repeats **until a final answer is ready**.

---

### **🔹 Adding Human in the Loop (HITL)**
To allow **human intervention before the AI generates a response**:
```python
memory = MemorySaver()
graph = builder.compile(interrupt_before=["assistant"], checkpointer=memory)
```
- **Interrupting before "assistant" runs** means a human can **review or modify the AI’s decision** before proceeding.
- **MemorySaver** keeps track of execution state.

---

### **🔹 Visualizing the Workflow**
Finally, the **LangGraph execution flow is displayed**:
```python
from IPython.display import Image, display
display(Image(graph.get_graph().draw_mermaid_png()))
```
This generates a **graph diagram** showing how the AI interacts with tools and humans.

---

## **3. Summary**
### **✅ Key Features**
1. **AI-Augmented Arithmetic Assistant**
   - Uses **Groq LLM (Qwen-2.5-32B)** to generate responses.
   - Can **call functions (tools) for arithmetic operations**.
  
2. **LangGraph Workflow for AI Decision-Making**
   - AI **determines whether a tool is needed** before responding.
   - Uses **state management and conditional routing**.

3. **Human-in-the-Loop (HITL) for Decision Control**
   - AI **pauses before execution**, allowing a human to review or modify its decision.
   - Useful for **debugging, approvals, and editing** AI responses.

4. **Graph Visualization**
   - Shows how AI **moves through different steps** (assistant, tools, human intervention).

---

## **4. How This Can Be Used**
- **AI Assistants** 🧑‍💻: Create an **AI that assists with decision-making** but allows human oversight.
- **Finance & Compliance** 💰: Automate **financial calculations** while requiring **human approval** for critical steps.
- **Debugging AI Models** 🔍: AI-generated responses **can be inspected and corrected** in real time.
- **Medical or Legal Applications** ⚖️: AI **suggests actions**, but a human must **review before proceeding**.

---

### **🚀 Next Steps (Potential Improvements)**
✅ **Expand Toolset**: Add more AI functions (e.g., advanced math, NLP tasks).  
✅ **Enhance HITL Mechanism**: Provide a UI for **human intervention** (e.g., accept/modify AI decisions).  
✅ **Multi-Model Support**: Combine **Groq AI with OpenAI (GPT-4o)** for comparison.  
✅ **Logging & Error Handling**: Ensure AI actions are **traceable and correctable**.  

---

## **5. Conclusion**
This notebook is a **powerful example of Human-in-the-Loop AI orchestration** using **LangGraph**. It demonstrates how AI can **assist with decision-making**, while allowing human intervention **before final execution**. This approach enhances **accuracy, transparency, and safety** in AI-driven workflows. 🚀