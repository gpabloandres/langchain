import os
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain import hub
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import Tool
load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)


def get_current_time(*args, **kwargs):
    """Returns the current time in H:MM AM/PM format."""
    import datetime

    now = datetime.datetime.now()
    return now.strftime("%I:%M %p")


def search_wikipedia(query):
    """Searches Wikipedia and returns the summary of the first result."""
    from wikipedia import summary

    try:
        # Limit to two sentences for brevity
        return summary(query, sentences=2)
    except:
        return "I couldn't find any information on that."
    

tools = [
    Tool(
        name="Time",
        func=get_current_time,
        description="Useful for when you need to know the current time.",
    ),
    Tool(
        name="Wikipedia",
        func=search_wikipedia,
        description="Useful for when you need to know information about a topic.",
    ),
]


prompt = hub.pull("hwchase17/structured-chat-agent")

memory = ConversationBufferMemory(
    memory_key="chat_history", return_messages=True)

agent = create_structured_chat_agent(llm=llm, tools=tools, prompt=prompt)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
    memory=memory,  
    handle_parsing_errors=True,
)

initial_message = "You are an AI assistant that can provide helpful answers using available tools.\nIf you are unable to answer, you can use the following tools: Time and Wikipedia."
memory.chat_memory.add_message(SystemMessage(content=initial_message))

while True:
    user_input = input("User: ")
    if user_input.lower() == "exit":
        break
    
    memory.chat_memory.add_message(HumanMessage(content=user_input))
    
    response = agent_executor.invoke({"input": user_input})
    print("Bot:", response["output"])
    
    memory.chat_memory.add_message(AIMessage(content=response["output"]))