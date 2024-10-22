from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

"""
messages = [
    SystemMessage(content="Resuelve el siguiente problema matemático"),
    HumanMessage(content="Cuánto es 81 dividido 9?"),
]
"""
messages = [
    SystemMessage(content="Resuelve el siguiente problema matemático"),
    HumanMessage(content="Cuánto es 81 dividido 9?"),
    AIMessage(content="81 dividido 9 es 9"),
    HumanMessage(content="Cuánto es 10 multiplicado por 5?")
]

result = model.invoke(messages)
print(f"Respuesta de la IA: {result.content}")