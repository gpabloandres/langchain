import os
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain import hub
from langchain.agents import (
    AgentExecutor,
    create_react_agent,
)
from langchain_core.tools import Tool

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

def get_current_time(*args, **kwargs):
    """Returns the current time in H:MM AM/PM format."""
    import datetime 
    now = datetime.datetime.now()
    return now.strftime("%I:%M %p")

tools = [
    Tool(
        name="Time",
        func=get_current_time,
        description="Useful for when you need to know the current time",
    ),
]

prompt = hub.pull("hwchase17/react")

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
    stop_sequence=True,
)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
)

response = agent_executor.invoke({"input": "What time is it?"})

print("response:", response)
