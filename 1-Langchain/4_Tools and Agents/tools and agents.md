# **📌 Tools & Agents in LangChain**
In **LangChain**, **tools** and **agents** work together to create **intelligent AI-powered applications** that can **search the web, query databases, retrieve documents, call APIs, and more**.

---

## **🚀 What Are Tools in LangChain?**
**Tools** are external functions or APIs that an **AI agent can call** to perform specific actions.

✅ **AI doesn’t just respond to text** → It can **search, fetch data, run calculations, call APIs, etc.**  
✅ **Agents decide when to use tools** → The AI can **dynamically pick the right tool** for a task.

---

## **🛠 Commonly Used Tools in LangChain**
| **Tool** | **Function** | **Example Use Case** |
|---------|------------|------------------|
| **SerpAPI** | Web search | AI chatbot with real-time Google search |
| **Wikipedia API** | Retrieve Wikipedia articles | AI research assistant |
| **SQLDatabaseTool** | Query SQL databases | AI-powered data analytics |
| **Python REPL** | Run Python code | AI-powered code execution |
| **Zapier Tool** | Connect to 5,000+ apps | AI task automation |
| **Arxiv Tool** | Fetch research papers | AI academic assistant |
| **FAISS Vector Store** | Retrieve similar documents | Retrieval-Augmented Generation (RAG) |

---

## **📌 Example: Creating a Custom Tool**
Let’s define a simple **tool** that fetches stock prices.

```python
from langchain.tools import Tool

# Define a stock price lookup function
def get_stock_price(stock_symbol):
    stock_prices = {"AAPL": 150, "GOOG": 2800, "TSLA": 700}
    return f"Stock price of {stock_symbol}: {stock_prices.get(stock_symbol, 'Unknown')}"

# Convert it into a LangChain tool
stock_price_tool = Tool(
    name="StockPriceLookup",
    func=get_stock_price,
    description="Fetch the stock price of a company by its stock symbol."
)

# Use the tool
print(stock_price_tool.run("AAPL"))  # Output: Stock price of AAPL: 150
```
✔ **Now, AI can use this tool dynamically!**  

---

## **🚀 What Are Agents in LangChain?**
**Agents** are **decision-making AI systems** that decide **which tools to use and when**.

✅ **Autonomous AI** → The agent **chooses actions** instead of following fixed steps.  
✅ **LLM-Driven** → Uses **GPT-4, Claude, or other LLMs** to make decisions.  
✅ **Multi-Step Reasoning** → Can **search, retrieve, and process** information dynamically.

---

## **🛠 Types of Agents in LangChain**
| **Agent Type** | **Description** | **Best Use Case** |
|--------------|----------------|------------------|
| **ZeroShotAgent** | AI decides tool usage without training | General-purpose AI assistants |
| **ReactAgent** | Uses reasoning steps before tool execution | Complex decision-making |
| **ConversationalAgent** | Remembers past conversations | AI chatbots |
| **Self-Ask Agent** | Asks follow-up questions before answering | Research assistants |

---

## **📌 Example: Building an AI Agent with Tools**
Let’s create an **AI agent** that **retrieves stock prices** and **performs calculations**.

### **🔹 Step 1: Install Dependencies**
```bash
pip install langchain openai
```

---

### **🔹 Step 2: Define Tools**
```python
from langchain.tools import Tool

# Define stock lookup function
def get_stock_price(stock_symbol):
    stock_prices = {"AAPL": 150, "GOOG": 2800, "TSLA": 700}
    return f"Stock price of {stock_symbol}: {stock_prices.get(stock_symbol, 'Unknown')}"

# Define a simple calculator
def calculate(expression):
    return eval(expression)

# Convert into tools
stock_tool = Tool(name="StockPriceLookup", func=get_stock_price, description="Get stock price.")
calc_tool = Tool(name="Calculator", func=calculate, description="Perform mathematical calculations.")
```

---

### **🔹 Step 3: Create an AI Agent**
```python
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType

# Initialize LLM
llm = ChatOpenAI(model_name="gpt-4", temperature=0)

# Create an agent with tools
agent = initialize_agent(
    tools=[stock_tool, calc_tool],  # Add tools here
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # AI can pick tools dynamically
    verbose=True
)

# Ask the agent a question
response = agent.run("What is the stock price of AAPL? Then multiply it by 2.")
print(response)
```
✔ **AI first fetches stock price, then performs the calculation automatically!** 🎉

---

## **📌 How Agents Work Internally**
1️⃣ **User asks a question** → `"What is the stock price of AAPL? Then multiply it by 2."`  
2️⃣ **AI chooses the right tool** → `"StockPriceLookup"`  
3️⃣ **AI retrieves stock price** → `"150"`  
4️⃣ **AI decides next action** → Calls `"Calculator"` with `"150 * 2"`  
5️⃣ **Final Answer** → `"300"`  

✔ **AI performs **multi-step reasoning** without predefined logic!**

---

## **🚀 Advanced Agent Capabilities**
✅ **Memory Integration** – Use **Redis or FAISS** to store chat history.  
✅ **RAG Support** – Agents can retrieve **external knowledge** before responding.  
✅ **Parallel Execution** – AI can **execute multiple tools simultaneously**.  
✅ **Custom Toolkits** – Define **industry-specific AI agents** (e.g., finance, legal, healthcare).  

---

## **🔥 Which Should You Use?**
| **Feature** | **Tools** | **Agents** |
|------------|---------|---------|
| **Executes a function/API** | ✅ | ✅ |
| **Dynamically chooses actions** | ❌ | ✅ |
| **Performs multi-step reasoning** | ❌ | ✅ |
| **Stores & retrieves knowledge** | ❌ | ✅ |
| **Good for predefined tasks** | ✅ | ❌ |
| **Good for open-ended AI** | ❌ | ✅ |

✔ **Use Tools** → When you need **specific functionalities** (e.g., stock price lookup, API calls).  
✔ **Use Agents** → When AI **needs to reason, decide, and call tools dynamically**.

---

## **🎯 Final Thoughts**
🚀 **Tools** = Functions AI can call 📞  
🚀 **Agents** = AI **decides** which tools to use 🤖  

