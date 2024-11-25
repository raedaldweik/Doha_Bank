import streamlit as st
from langchain_community.agent_toolkits import create_sql_agent
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Set up OpenAI API key from environment variable
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("API key not found. Please check your .env file.")
else:
    os.environ["OPENAI_API_KEY"] = api_key  # Set API key as environment variable for OpenAI

# Database setup
engine = create_engine("sqlite:///doha_bank.db")
db = SQLDatabase(engine=engine)
llm = ChatOpenAI(model="gpt-4o-mini")
agent_executor = create_sql_agent(llm, db=db, agent_type="openai-tools", verbose=True)

# Data dictionary for context
data_dictionary = """
| Column Name              | Description                                                                 |
|--------------------------|-----------------------------------------------------------------------------|
| Customer ID              | Unique identifier for each customer                                         |
| Customer Name            | Name of the customer (e.g., business name)                                 |
| Customer Type            | Type of customer (e.g., Micro/Small Enterprise, Medium Enterprise, Corporate) |
| City                     | City where the customer is located                                         |
| Longitude                | Geographic longitude of the customer's location                           |
| Latitude                 | Geographic latitude of the customer's location                            |
| Annual Turnover          | Annual turnover of the customer in monetary units                         |
| Total Exposure           | Total financial exposure associated with the customer                     |
| Total Limit              | Credit limit assigned to the customer                                      |
| Industry                 | Industry or sector the customer belongs to                                |
| Customer Seniority       | Duration (in months) of the customer's relationship with the bank          |
| Watchlist                | Watchlist classification for the customer                                 |
| EWS Action               | Recommended action based on Early Warning System analysis                 |
| Final_EWS_Score          | Final Early Warning System score for the customer                         |
| Predictive Indicator     | Predictive metric indicating potential risk                               |
| Predicted Risk Category  | Categorized risk level (e.g., Green, Amber, Red) based on predictions      |
| Predicted Risk Level     | Numerical risk level derived from predictive analysis                     |
| Status                   | Current status of the customer's action plan (e.g., Action Taken)         |
"""


# Streamlit UI setup
st.title("Doha Bank Generative AI Assistant")
st.write("Ask me anything!")

# Chatbot conversation state
if "conversation" not in st.session_state:
    st.session_state.conversation = []

# User input
user_input = st.text_input("You:", key="user_input")

if user_input:
    # Add the data dictionary to the input for better context
    input_text = f"Refer to the following data dictionary for context:\n\n{data_dictionary}\n\n{user_input}"
    # Query the RAG model
    result = agent_executor.invoke({"input": input_text})["output"]
    # Append conversation history
    st.session_state.conversation.append(("User", user_input))
    st.session_state.conversation.append(("Bot", result))
    user_input = ""  # Clear input after submission

# Display conversation history in a container with autoscroll enabled
with st.container():
    for speaker, text in st.session_state.conversation:
        if speaker == "User":
            st.write(f"**You:** {text}")
        else:
            st.write(f"**Bot:** {text}")
    # Automatically scrolls to the latest conversation entry
    st_autoscroll = True
