### **🚀 Explanation of Routing-Based Workflow with Structured LLM Output**
This script demonstrates **a routing-based workflow** using **LangChain, Pydantic, and LangGraph** to classify user input and invoke an appropriate function dynamically. 

---
## **📌 How the Workflow Works**
1. **User Input** → The user provides a request (e.g., *"Write me a joke about cats."*).
2. **AI Routing Decision** → An **LLM model classifies the request** as either a **poem, story, or joke**.
3. **Structured Output** → The LLM outputs a **structured response** specifying which function should handle the request.
4. **Dynamic Routing** → The workflow **automatically routes** the request to the correct function:
   - `llm_call_1`: Writes a **story**.
   - `llm_call_2`: Writes a **joke**.
   - `llm_call_3`: Writes a **poem**.
5. **Final Output** → The appropriate function processes the request and returns the result.

---

## **🔹 Key Components in the Code**
### **1️⃣ Importing Required Libraries**
```python
from typing_extensions import Literal
from typing import TypedDict
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.chat_models import ChatOpenAI  # Assuming using OpenAI LLM
from langgraph.graph import StateGraph  # Import the StateGraph
from IPython.display import Image, display  # Import for visualization
```
- **`ChatOpenAI`** → Used to initialize GPT-4 as the LLM.
- **`BaseModel`, `Field`** → Define structured models for AI-generated routing.
- **`TypedDict`** → Defines state objects used in the workflow.
- **`StateGraph`** → Manages workflow execution.
- **`Image, display`** → For visualizing the workflow.

---

### **2️⃣ Initialize the LLM (GPT-4)**
```python
llm = ChatOpenAI(model_name="gpt-4")  # Using OpenAI's GPT-4
```
- The **LLM is initialized** and will be used to generate structured responses.

---

### **3️⃣ Define Routing Schema (Structured LLM Output)**
```python
class Route(BaseModel):
    step: Literal["poem", "story", "joke"] = Field(
        ..., description="The next step in the routing process"
    )
```
- **Defines structured AI output**:
  - The AI can only **output one of three valid options**: `"poem"`, `"story"`, or `"joke"`.
  - The structured output **forces the AI to return a valid routing decision**.

---

### **4️⃣ Augment the LLM to Return Structured Output**
```python
router = llm.with_structured_output(Route)
```
- This **modifies the LLM** so it returns structured JSON-like responses **instead of free text**.

---

### **5️⃣ Define Workflow State**
```python
class State(TypedDict):
    input: str
    decision: str
    output: str
```
- This dictionary holds **three values**:
  - `"input"` → The user's original request.
  - `"decision"` → The AI-determined category (`"poem"`, `"story"`, or `"joke"`).
  - `"output"` → The final generated response.

---

### **6️⃣ Define Nodes (Processing Functions)**
Each function corresponds to an action the AI can take.

#### **Story Generator**
```python
def llm_call_1(state: State):
    """Write a story"""
    result = llm.invoke(state["input"])
    return {"output": result.content}
```

#### **Joke Generator**
```python
def llm_call_2(state: State):
    """Write a joke"""
    print("LLM call 2 is called")
    result = llm.invoke(state["input"])
    return {"output": result.content}
```

#### **Poem Generator**
```python
def llm_call_3(state: State):
    """Write a poem"""
    result = llm.invoke(state["input"])
    return {"output": result.content}
```
- Each function **calls the LLM** with the user’s input and **returns generated content**.

---

### **7️⃣ Define Routing Logic**
```python
def llm_call_router(state: State):
    """Route the input to the appropriate node"""
    decision = router.invoke(
        [
            SystemMessage(content="Route the input to story, joke, or poem based on the user's request."),
            HumanMessage(content=state["input"]),
        ]
    )
    return {"decision": decision.step}
```
- This function **calls the AI** to **decide** which function to use (`poem`, `story`, or `joke`).
- The AI receives **system instructions** + **user input** and returns a **structured decision**.

---

### **8️⃣ Define Conditional Routing**
```python
def route_decision(state: State):
    if state["decision"] == "story":
        return "llm_call_1"
    elif state["decision"] == "joke":
        return "llm_call_2"
    elif state["decision"] == "poem":
        return "llm_call_3"
```
- The AI’s decision **determines which function to call**.

---

### **9️⃣ Build the Workflow Graph**
```python
router_builder = StateGraph(State)

# Add nodes
router_builder.add_node("llm_call_1", llm_call_1)
router_builder.add_node("llm_call_2", llm_call_2)
router_builder.add_node("llm_call_3", llm_call_3)
router_builder.add_node("llm_call_router", llm_call_router)
```
- **StateGraph** is used to **create and manage the workflow**.
- Nodes (`llm_call_1`, `llm_call_2`, `llm_call_3`, `llm_call_router`) represent **steps in execution**.

---

### **🔟 Define Workflow Edges (Connections)**
```python
# Add edges to connect nodes
router_builder.add_edge("start", "llm_call_router")
router_builder.add_conditional_edges(
    "llm_call_router",
    route_decision,
    {
        "llm_call_1": "llm_call_1",
        "llm_call_2": "llm_call_2",
        "llm_call_3": "llm_call_3",
    },
)
router_builder.add_edge("llm_call_1", "end")
router_builder.add_edge("llm_call_2", "end")
router_builder.add_edge("llm_call_3", "end")
```
- **Connects nodes together**, so input flows from **start → router → correct function → end**.

---

### **🔹 11. Compile & Visualize Workflow**
```python
router_workflow = router_builder.compile()
display(Image(router_workflow.get_graph().draw_mermaid_png()))
```
- **Compiles the workflow** and **generates a visual representation**.

---

### **🔹 12. Execute the Workflow**
```python
state = router_workflow.invoke({"input": "Write me a joke about cats"})
print(state["output"])
```
- **Invokes the workflow with user input**.
- The AI decides which function to call.
- The correct function is executed and **outputs a generated response**.

---

## **🚀 Summary of Workflow Execution**
1️⃣ **User input:** `"Write me a joke about cats"`  
2️⃣ **AI routes input:** `"joke"`  
3️⃣ **Workflow calls `llm_call_2()`**  
4️⃣ **LLM generates a joke**  
5️⃣ **Final response is displayed**  

Example Output:
```
😹 Why don't cats play poker in the jungle? 
Too many cheetahs! 🐆
```

---

## **📌 Why Use Routing-Based Workflows?**
✅ **Dynamically routes tasks** → No manual classification required.  
✅ **Uses AI-generated structured output** → Ensures correct decisions.  
✅ **Scales easily** → More tasks (e.g., songs, essays) can be added.  
✅ **Automates workflows** → Can integrate with chatbots, APIs, or web apps.  

---
