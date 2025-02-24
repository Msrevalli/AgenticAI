# 📌 Import Required Libraries
import os
import random
from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict

from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# ✅ Load Environment Variables
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# ✅ Define Chat State using TypedDict
class State(TypedDict):
    messages: Annotated[list, add_messages]  # Stores chat messages

# ✅ Initialize LangGraph Execution Model
graph_builder = StateGraph(State)

# ✅ Load Groq LLM Model (gemma2-9b-it)
llm = ChatGroq(model="gemma2-9b-it")

# ✅ Define the Chatbot Function
def chatbot(state: State):
    """
    Processes user input and returns AI-generated messages.
    """
    return {"messages": [llm.invoke(state["messages"])]}

# ✅ Build the AI Graph
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")  # Start -> chatbot
graph_builder.add_edge("chatbot", END)  # chatbot -> End
graph = graph_builder.compile()

# ✅ Function to Stream AI Responses
def stream_graph_updates(user_input: str):
    """
    Streams AI-generated responses in real-time.
    """
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)

# ✅ Start Chatbot Interaction
def run_chatbot():
    """
    Runs the chatbot in an interactive terminal loop.
    """
    print("\n🤖 LangGraph Chatbot (Type 'exit' to quit)\n")
    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye! 👋")
                break

            stream_graph_updates(user_input)
        except:
            # Fallback for environments without interactive input
            user_input = "What do you know about LangGraph?"
            print("User: " + user_input)
            stream_graph_updates(user_input)
            break

# ✅ Run the Chatbot
if __name__ == "__main__":
    run_chatbot()
