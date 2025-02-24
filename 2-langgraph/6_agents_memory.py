# 📌 Import Required Libraries
import os
from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, SystemMessage, AnyMessage
from langgraph.graph.message import add_messages

# ✅ Load Environment Variables
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# ✅ Initialize Groq's LLM (DeepSeek 70B)
llm = ChatGroq(model="deepseek-r1-distill-llama-70b")

# ✅ Define AI Callable Tools
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

def divide(a: int, b: int) -> float:
    """Divide two numbers."""
    return a / b

# ✅ Bind the Tools to the AI Model
tools = [add, multiply, divide]
llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False)

# ✅ Define AI State using TypedDict
class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

# ✅ Define AI Processing Node
sys_msg = SystemMessage(content="You are a helpful assistant tasked with performing arithmetic on a set of inputs.")

def assistant(state: MessagesState):
    """
    AI processes user input and decides whether to call tools.
    """
    return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

# ✅ Build AI Graph Execution Model
builder = StateGraph(MessagesState)

# ➤ Add Nodes
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

# ➤ Define Execution Flow
builder.add_edge(START, "assistant")  # Start -> AI Processes Input
builder.add_conditional_edges("assistant", tools_condition)  # AI decides if tools are needed
builder.add_edge("tools", "assistant")  # Tools send results back to AI for further processing

# ✅ Add Memory for Multi-Turn Conversation Support
memory = MemorySaver()
react_graph = builder.compile(checkpointer=memory)

# ✅ Function to Process AI Responses
def process_input(user_input: str, thread_id: str):
    """
    Sends user input through the graph and returns AI responses, maintaining conversation history.
    """
    messages = [HumanMessage(content=user_input)]
    config = {"configurable": {"thread_id": thread_id}}
    
    response = react_graph.invoke({"messages": messages}, config)
    
    for msg in response['messages']:
        msg.pretty_print()

# ✅ Run Chatbot with Memory Support
if __name__ == "__main__":
    print("\n🤖 AI Chatbot with Memory (LangGraph) | Type 'exit' to quit\n")
    
    while True:
        try:
            thread_id = input("Enter thread ID (e.g., '1', '2', 'test') or type 'exit' to quit: ")
            if thread_id.lower() in ["quit", "exit", "q"]:
                print("Goodbye! 👋")
                break
            
            while True:
                user_input = input(f"User ({thread_id}): ")
                if user_input.lower() in ["quit", "exit", "q"]:
                    print(f"Conversation thread {thread_id} ended.")
                    break

                process_input(user_input, thread_id)

        except Exception as e:
            print("Error:", str(e))
            break
