import streamlit as st
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain.agents import AgentType, initialize_agent
from langchain_community.tools.google_finance import GoogleFinanceQueryRun
from langchain_community.utilities.google_finance import GoogleFinanceAPIWrapper

import os


# Set API keys
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]


# Initialize LLM
llm = ChatGroq(model="qwen-2.5-32b")

# Initialize Tools
tool = GoogleFinanceQueryRun(api_wrapper=GoogleFinanceAPIWrapper(serp_api_key=st.secrets["SERPAPI_API_KEY"]))

tools = [tool]

# Initialize the agent
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)

# Define State
class State(TypedDict):
    investment_asset: str
    technical_insights: str
    fundamental_insights: str
    sentiment_insights: str
    risk_evaluation: str
    final_report: str

# Define AI Functions
def technical_analysis(state: State):
    """Runs Technical Analysis (RSI, MACD, Bollinger Bands)"""
    prompt = f"""
    Perform a comprehensive technical analysis on {state['investment_asset']} with the following structure:
    1. Current Price Trend: Identify whether the asset is in an uptrend, downtrend, or trading sideways
    2. Support and Resistance Levels: Identify key price levels
    3. RSI Analysis: Current RSI value and whether the asset is overbought or oversold
    4. MACD Analysis: Current signal and potential crossovers
    5. Moving Averages: Relationship between short-term and long-term moving averages
    6. Volume Analysis: Recent volume trends and what they indicate
    7. Technical Outlook: Overall technical forecast based on indicators
    
    Be specific with numbers where possible and explain the technical significance.
    """
    msg = agent.invoke(prompt)
    return {"technical_insights": msg["output"]}

def fundamental_analysis(state: State):
    """Runs Fundamental Analysis (P/E Ratio, Revenue, Earnings)"""
    prompt = f"""
    Perform a detailed fundamental analysis on {state['investment_asset']} covering:
    1. Valuation Metrics: P/E ratio, P/B ratio, P/S ratio compared to industry averages
    2. Financial Health: Debt-to-equity ratio, current ratio, quick ratio
    3. Growth Metrics: Revenue growth rate, earnings growth rate, projected growth
    4. Profitability: Profit margins, ROE, ROA, and their trends
    5. Cash Flow Analysis: Free cash flow, operating cash flow trends
    6. Dividend Analysis (if applicable): Yield, payout ratio, dividend growth
    7. Fundamental Outlook: Overall assessment based on fundamentals
    
    For cryptocurrencies, adapt metrics to include market cap, trading volume, network metrics, and development activity.
    For real estate, focus on cap rates, NOI, vacancy rates, and regional market trends.
    """
    msg = agent.invoke(prompt)
    return {"fundamental_insights": msg["output"]}

def sentiment_analysis(state: State):
    """Runs Sentiment Analysis (News, Social Media, Market Trends)"""
    prompt = f"""
    Analyze the current market sentiment for {state['investment_asset']} by examining:
    1. Recent News Coverage: Summarize major news and its impact (positive/negative)
    2. Social Media Sentiment: General tone on Twitter, Reddit, and other platforms
    3. Analyst Opinions: Recent analyst ratings, price targets, and consensus
    4. Institutional Interest: Recent institutional buying or selling activity
    5. Retail Investor Sentiment: Retail investor interest and sentiment trends
    6. Market Narratives: Dominant narratives or stories surrounding this asset
    7. Sentiment Outlook: Overall sentiment assessment and potential market psychology factors
    
    Provide specific examples of recent sentiment drivers where possible.
    """
    msg = agent.invoke(prompt)
    return {"sentiment_insights": msg["output"]}

def risk_analysis(state: State):
    """Runs Risk Assessment (Volatility, Market Conditions)"""
    prompt = f"""
    Conduct a comprehensive risk assessment for {state['investment_asset']} by analyzing:
    1. Volatility Metrics: Historical volatility, beta (for stocks), and comparison to benchmarks
    2. Downside Risk: Maximum drawdown history, potential downside scenarios
    3. Correlation: Correlation with broader market and diversification potential
    4. Liquidity Risk: Trading volume, bid-ask spreads, and liquidity concerns
    5. Regulatory/Legal Risks: Pending regulations or legal challenges
    6. Industry-Specific Risks: Competitive threats, disruption potential, industry headwinds
    7. Macroeconomic Sensitivity: How economic factors (interest rates, inflation) affect this asset
    8. Risk Mitigation Strategies: Potential hedging or risk management approaches
    
    Provide a risk rating (Low/Medium/High) with justification.
    """
    msg = agent.invoke(prompt)
    return {"risk_evaluation": msg["output"]}

def aggregator(state: State):
    """Combines all investment insights into a final report"""
    prompt = f"""
    Create a comprehensive investment report for {state['investment_asset']} by synthesizing:
    
    Technical Analysis: {state['technical_insights']}
    
    Fundamental Analysis: {state['fundamental_insights']}
    
    Sentiment Analysis: {state['sentiment_insights']}
    
    Risk Assessment: {state['risk_evaluation']}
    
    Based on the above analyses, provide:
    1. Investment Thesis: Core reasoning for bullish or bearish outlook
    2. Key Strengths: Most compelling reasons to invest
    3. Key Concerns: Most significant risks or red flags
    4. Time Horizon: Suitable investment timeframe (short, medium, long-term)
    5. Price Targets: Potential upside and downside scenarios with percentages
    6. Final Recommendation: Clear buy/hold/sell recommendation with confidence level
    7. Suggested Position Sizing: Based on the risk profile of this investment
    8. Alternative Investments: Similar assets that might be worth considering
    
    Format as a clean, professional investment report with clear sections.
    """
    msg = agent.invoke(prompt)
    return {"final_report": msg["output"]}

def asset(state):
    return state
# Define Graph
investment_builder = StateGraph(State)

# Add Nodes
investment_builder.add_node("asset", asset)
investment_builder.add_node("technical_analysis", technical_analysis)
investment_builder.add_node("fundamental_analysis", fundamental_analysis)
investment_builder.add_node("sentiment_analysis", sentiment_analysis)
investment_builder.add_node("risk_analysis", risk_analysis)
investment_builder.add_node("aggregator", aggregator)

# Connect Nodes
investment_builder.add_edge(START, "asset")
investment_builder.add_edge("asset", "technical_analysis")
investment_builder.add_edge("asset", "technical_analysis")
investment_builder.add_edge("asset", "fundamental_analysis")
investment_builder.add_edge("asset", "sentiment_analysis")
investment_builder.add_edge("asset", "risk_analysis")

investment_builder.add_edge("technical_analysis", "aggregator")
investment_builder.add_edge("fundamental_analysis", "aggregator")
investment_builder.add_edge("sentiment_analysis", "aggregator")
investment_builder.add_edge("risk_analysis", "aggregator")

investment_builder.add_edge("aggregator", END)

# Compile Workflow
investment_workflow = investment_builder.compile()

# Streamlit UI
st.title("AI-Powered Investment Analyzer")
st.write("Analyze stocks, crypto, or real estate assets using AI-driven technical, fundamental, sentiment, and risk analysis.")

# User Input
asset_text = st.text_input("Enter an asset (e.g., Tesla, Bitcoin, NYC Real Estate)")
with st.sidebar:
    st.subheader("Workflow Diagram")

    # ✅ Generate Mermaid Workflow Diagram
    mermaid_diagram = investment_workflow.get_graph().draw_mermaid_png()

    # ✅ Save and Display the Image in Sidebar
    image_path = "workflow_diagram.png"
    with open(image_path, "wb") as f:
        f.write(mermaid_diagram)

    st.image(image_path, caption="Workflow Execution")


if st.button("Analyze Investment"):
    if asset_text:
        state = investment_workflow.invoke({"investment_asset": asset_text})

        # Display Investment Report
        st.subheader("Investment Report")
        st.markdown(state["final_report"])
    else:
        st.warning("Please enter an asset to analyze.")

    st.markdown("### 🔗 Powered by LangGraph with Parallelization Workflow")
    
