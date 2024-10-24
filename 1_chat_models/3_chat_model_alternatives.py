from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

load_dotenv()

model = ChatAnthropic(model="claude-3-opus-20240229", temperature=0.3)

messages = [
    SystemMessage(content="Resuelve el siguiente problema matemático"),
    HumanMessage(content="Cuánto es 81 dividido 9?"),
]

result = model.invoke(messages)

print(f"Respuesta desde Anthropic: {result.content}")