import os
from typing import Type
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain import hub
from langchain.agents import AgentExecutor, create_tool_calling_agent
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

class SimpleSearchInput(BaseModel):
    query: str = Field(description="should be a search query")

class MultiplyNumbersArgs(BaseModel):
    x: float = Field(description="First number to multiply")
    y: float = Field(description="Second number to multiply")

class SimpleSearchTool(BaseTool):
    name: str = "simple_search"
    description: str = "useful for when you need to answer questions about current events"
    args_schema: Type[BaseModel] = SimpleSearchInput

    def _run(
        self,
        query: str,
    ) -> str:
        """Use the tool."""
        from tavily import TavilyClient

        api_key = os.getenv("TAVILY_API_KEY")
        client = TavilyClient(api_key=api_key)
        results = client.search(query=query)
        return f"Search results for: {query}\n\n\n{results}\n"

class MultiplyNumbersTool(BaseTool):
    name: str = "multiply_numbers"
    description: str = "useful for multiplying two numbers"
    args_schema: Type[BaseModel] = MultiplyNumbersArgs

    def _run(
        self,
        x: float,
        y: float,
    ) -> str:
        """Use the tool."""
        result = x * y
        return f"The product of {x} and {y} is {result}"

tools = [
    SimpleSearchTool(),
    MultiplyNumbersTool(),
]

prompt = hub.pull("hwchase17/openai-tools-agent")

agent = create_tool_calling_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
)

response = agent_executor.invoke({"input": "Search for Apple Intelligence"})
print("Response for 'Search for LangChain updates':", response)

response = agent_executor.invoke({"input": "Multiply 10 and 20"})
print("Response for 'Multiply 10 and 20':", response)