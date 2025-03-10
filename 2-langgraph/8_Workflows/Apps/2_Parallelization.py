import streamlit as st
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain.agents import AgentType, initialize_agent
from langchain_community.tools.google_finance import GoogleFinanceQueryRun
from langchain_community.utilities.google_finance import GoogleFinanceAPIWrapper
import time
import os
import traceback


# Add page configuration
st.set_page_config(
    page_title="AI-Powered Investment Analyzer",
    page_icon="📊",
    layout="wide"
)

# Set API keys
try:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except Exception as e:
    st.error("Error setting GROQ API Key. Please check your secrets.")
    st.stop()

# Initialize LLM with error handling
@st.cache_resource
def get_llm():
    try:
        return ChatGroq(model="qwen-2.5-32b", temperature=0.2)
    except Exception as e:
        st.error(f"Error initializing LLM: {str(e)}")
        st.stop()

# Initialize Tools with error handling
@st.cache_resource
def get_tools():
    try:
        tool = GoogleFinanceQueryRun(api_wrapper=GoogleFinanceAPIWrapper(serp_api_key=st.secrets["SERPAPI_API_KEY"]))
        return [tool]
    except Exception as e:
        st.error(f"Error initializing tools: {str(e)}")
        st.stop()

# Initialize the agent with error handling
@st.cache_resource
def get_agent():
    llm = get_llm()
    tools = get_tools()
    try:
        return initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    except Exception as e:
        st.error(f"Error initializing agent: {str(e)}")
        st.stop()

# Define State
class State(TypedDict):
    investment_asset: str
    technical_insights: str
    fundamental_insights: str
    sentiment_insights: str
    risk_evaluation: str
    final_report: str
    error: str

# Define AI Functions with error handling
def technical_analysis(state: State):
    """Runs Technical Analysis (RSI, MACD, Bollinger Bands)"""
    try:
        agent = get_agent()
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
    except Exception as e:
        error_msg = f"Error during technical analysis: {str(e)}"
        return {"technical_insights": f"⚠️ {error_msg}", "error": error_msg}

def fundamental_analysis(state: State):
    """Runs Fundamental Analysis (P/E Ratio, Revenue, Earnings)"""
    try:
        agent = get_agent()
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
    except Exception as e:
        error_msg = f"Error during fundamental analysis: {str(e)}"
        return {"fundamental_insights": f"⚠️ {error_msg}", "error": error_msg}

def sentiment_analysis(state: State):
    """Runs Sentiment Analysis (News, Social Media, Market Trends)"""
    try:
        agent = get_agent()
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
    except Exception as e:
        error_msg = f"Error during sentiment analysis: {str(e)}"
        return {"sentiment_insights": f"⚠️ {error_msg}", "error": error_msg}

def risk_analysis(state: State):
    """Runs Risk Assessment (Volatility, Market Conditions)"""
    try:
        agent = get_agent()
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
    except Exception as e:
        error_msg = f"Error during risk analysis: {str(e)}"
        return {"risk_evaluation": f"⚠️ {error_msg}", "error": error_msg}

def aggregator(state: State):
    """Combines all investment insights into a final report"""
    try:
        # Check if any analysis had errors
        if state.get("error"):
            return {"final_report": f"⚠️ Some analyses encountered errors. Please check individual reports for details."}
            
        agent = get_agent()
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
    except Exception as e:
        error_msg = f"Error during report aggregation: {str(e)}"
        return {"final_report": f"⚠️ {error_msg}"}

def asset(state):
    return state

# Define Graph
def create_workflow():
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
    investment_builder.add_edge("asset", "fundamental_analysis")
    investment_builder.add_edge("asset", "sentiment_analysis")
    investment_builder.add_edge("asset", "risk_analysis")

    investment_builder.add_edge("technical_analysis", "aggregator")
    investment_builder.add_edge("fundamental_analysis", "aggregator")
    investment_builder.add_edge("sentiment_analysis", "aggregator")
    investment_builder.add_edge("risk_analysis", "aggregator")

    investment_builder.add_edge("aggregator", END)

    # Compile Workflow
    return investment_builder.compile()

# Streamlit UI
st.title("AI-Powered Investment Analyzer")
st.write("Analyze stocks, crypto, or real estate assets using AI-driven technical, fundamental, sentiment, and risk analysis.")

# User Input
asset = st.text_input("Enter an asset (e.g., Tesla, Bitcoin, NYC Real Estate)")

# Sidebar with workflow diagram
with st.sidebar:
    st.subheader("Workflow Diagram")
    
    try:
        # Generate workflow diagram
        investment_workflow = create_workflow()
        mermaid_diagram = investment_workflow.get_graph().draw_mermaid_png()

        # Save and Display the Image in Sidebar
        image_path = "workflow_diagram.png"
        with open(image_path, "wb") as f:
            f.write(mermaid_diagram)

        st.image(image_path, caption="Workflow Execution")
    except Exception as e:
        st.error(f"Error generating workflow diagram: {str(e)}")

# Add option for sequential or parallel mode
analysis_mode = st.radio(
    "Analysis Mode",
    ["Sequential (Safer)", "Parallel (Faster but may have errors)"],
    help="Sequential mode runs one analysis at a time. Parallel mode runs all analyses simultaneously."
)

if st.button("Analyze Investment"):
    if not asset:
        st.warning("Please enter an asset to analyze.")
        st.stop()
    
    # Progress tracking
    progress = st.progress(0)
    status_text = st.empty()
    
    try:
        # Initialize result dictionary
        results = {
            "investment_asset": asset,
            "technical_insights": "",
            "fundamental_insights": "",
            "sentiment_insights": "",
            "risk_evaluation": "",
            "final_report": ""
        }
        
        if analysis_mode == "Sequential (Safer)":
            # Sequential execution
            status_text.text("Performing technical analysis...")
            progress.progress(10)
            agent = get_agent()
            
            # Technical Analysis
            try:
                tech_prompt = f"""
                Perform a comprehensive technical analysis on {asset} with the following structure:
                1. Current Price Trend: Identify whether the asset is in an uptrend, downtrend, or trading sideways
                2. Support and Resistance Levels: Identify key price levels
                3. RSI Analysis: Current RSI value and whether the asset is overbought or oversold
                4. MACD Analysis: Current signal and potential crossovers
                5. Moving Averages: Relationship between short-term and long-term moving averages
                6. Volume Analysis: Recent volume trends and what they indicate
                7. Technical Outlook: Overall technical forecast based on indicators
                
                Be specific with numbers where possible and explain the technical significance.
                """
                msg = agent.invoke(tech_prompt)
                results["technical_insights"] = msg["output"]
            except Exception as e:
                results["technical_insights"] = f"⚠️ Error during technical analysis: {str(e)}"
            
            # Fundamental Analysis
            status_text.text("Performing fundamental analysis...")
            progress.progress(30)
            try:
                fund_prompt = f"""
                Perform a detailed fundamental analysis on {asset} covering:
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
                msg = agent.invoke(fund_prompt)
                results["fundamental_insights"] = msg["output"]
            except Exception as e:
                results["fundamental_insights"] = f"⚠️ Error during fundamental analysis: {str(e)}"
            
            # Sentiment Analysis
            status_text.text("Performing sentiment analysis...")
            progress.progress(50)
            try:
                sent_prompt = f"""
                Analyze the current market sentiment for {asset} by examining:
                1. Recent News Coverage: Summarize major news and its impact (positive/negative)
                2. Social Media Sentiment: General tone on Twitter, Reddit, and other platforms
                3. Analyst Opinions: Recent analyst ratings, price targets, and consensus
                4. Institutional Interest: Recent institutional buying or selling activity
                5. Retail Investor Sentiment: Retail investor interest and sentiment trends
                6. Market Narratives: Dominant narratives or stories surrounding this asset
                7. Sentiment Outlook: Overall sentiment assessment and potential market psychology factors
                
                Provide specific examples of recent sentiment drivers where possible.
                """
                msg = agent.invoke(sent_prompt)
                results["sentiment_insights"] = msg["output"]
            except Exception as e:
                results["sentiment_insights"] = f"⚠️ Error during sentiment analysis: {str(e)}"
            
            # Risk Analysis
            status_text.text("Performing risk analysis...")
            progress.progress(70)
            try:
                risk_prompt = f"""
                Conduct a comprehensive risk assessment for {asset} by analyzing:
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
                msg = agent.invoke(risk_prompt)
                results["risk_evaluation"] = msg["output"]
            except Exception as e:
                results["risk_evaluation"] = f"⚠️ Error during risk analysis: {str(e)}"
            
            # Final Report
            status_text.text("Generating final report...")
            progress.progress(90)
            try:
                agg_prompt = f"""
                Create a comprehensive investment report for {asset} by synthesizing:
                
                Technical Analysis: {results["technical_insights"]}
                
                Fundamental Analysis: {results["fundamental_insights"]}
                
                Sentiment Analysis: {results["sentiment_insights"]}
                
                Risk Assessment: {results["risk_evaluation"]}
                
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
                msg = agent.invoke(agg_prompt)
                results["final_report"] = msg["output"]
            except Exception as e:
                results["final_report"] = f"⚠️ Error generating final report: {str(e)}"
        else:
            # Parallel execution using the workflow
            status_text.text(f"Analyzing {asset} in parallel... This may take a few minutes.")
            investment_workflow = create_workflow()
            results = investment_workflow.invoke({"investment_asset": asset})
        
        # Display complete
        progress.progress(100)
        status_text.text("Analysis complete!")
        time.sleep(1)
        status_text.empty()
        
        # Display Executive Summary First
        st.subheader(f"Investment Analysis: {asset}")
        
        # Display Final Report in Main Area
        with st.expander("📊 Executive Summary", expanded=True):
            st.markdown(results["final_report"])
        
        # Display Detailed Analysis Reports in Expanders
        with st.expander("📈 Technical Analysis"):
            st.markdown(results["technical_insights"])
            
        with st.expander("💰 Fundamental Analysis"):
            st.markdown(results["fundamental_insights"])
            
        with st.expander("📰 Sentiment Analysis"):
            st.markdown(results["sentiment_insights"])
            
        with st.expander("⚠️ Risk Assessment"):
            st.markdown(results["risk_evaluation"])
        
    except Exception as e:
        st.error(f"An error occurred during analysis: {str(e)}")
        st.error(traceback.format_exc())

st.markdown("### 🔗 Powered by LangGraph with AI-Powered Investment Analysis 🚀")