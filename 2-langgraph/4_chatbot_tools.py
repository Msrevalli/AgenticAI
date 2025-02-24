# 📌 Import Required Libraries
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage

# ✅ Load Environment Variables
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# ✅ Initialize Groq's LLM (Qwen-2.5-32B)
llm = ChatGroq(model="qwen-2.5-32b")

# ✅ Define AI Callable Tools
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

# ✅ Bind the Tools to the AI Model
llm_with_tools = llm.bind_tools([multiply, add])

# ✅ Define AI Processing Node
def tool_calling_llm(state: MessagesState):
    """
    AI processes user input and decides whether to call tools.
    """
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# ✅ Build AI Graph Execution Model
builder = StateGraph(MessagesState)

# ➤ Add Nodes
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_node("tools", ToolNode([multiply, add]))

# ➤ Define Execution Flow
builder.add_edge(START, "tool_calling_llm")  # Start -> AI Processes Input
builder.add_conditional_edges("tool_calling_llm", tools_condition)  # AI decides if tools are needed
builder.add_edge("tools", END)  # If tools are called, end execution after usage

# ✅ Compile Graph
graph = builder.compile()

# ✅ Function to Process AI Responses
def process_input(user_input: str):
    """
    Sends user input through the graph and returns AI responses.
    """
    messages = [HumanMessage(content=user_input)]
    response = graph.invoke({"messages": messages})
    
    for msg in response['messages']:
        msg.pretty_print()

# ✅ Run Chatbot
if __name__ == "__main__":
    print("\n🤖 AI Chatbot (LangGraph) | Type 'exit' to quit\n")
    
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
