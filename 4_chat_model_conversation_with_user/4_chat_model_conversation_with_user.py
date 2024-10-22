import os
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

chat_history = []

system_message = SystemMessage(content="Tú eres un asistente AI ayudante")

chat_history.append(system_message)

while True:
    query = input("Tu consulta: ")
    if query.lower() == "salir":
        break
    chat_history.append(HumanMessage(content=query))
    
    result = model(chat_history)
    response = result.content
    chat_history.append(AIMessage(content=response))
    
    print(f"AI: {response}")
    
print("--- Message History ---")
print(chat_history)