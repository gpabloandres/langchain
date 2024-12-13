import os
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

# Lista que almacena la conversación completa entre el usuario y la IA.
chat_history = []

# Mensaje inicial que define el rol del asistente, en este caso, "un asistente AI ayudante".
system_message = SystemMessage(content="Tú eres un asistente AI ayudante")

# El mensaje del sistema se agrega al historial de chat.
chat_history.append(system_message)

# Inicia un bucle infinito para permitir al usuario enviar consultas a la IA.
while True:
    query = input("Tu consulta: ")
    
    # Si el usuario ingresa "salir", el bucle se detiene.
    if query.lower() == "salir":
        break
    
    # El mensaje del usuario se encapsula en un objeto HumanMessage y se agrega al historial de chat.
    chat_history.append(HumanMessage(content=query))
    
    # El modelo procesa el historial de mensajes y genera una respuesta.
    result = model(chat_history)
    
    # Extrae el contenido de la respuesta generada.
    response = result.content
    
    # La respuesta de la IA se encapsula como un objeto AIMessage y se agrega al historial.
    chat_history.append(AIMessage(content=response))
    
    # La respuesta generada por la IA se imprime en pantalla.
    print(f"AI: {response}")
    
# Al salir del bucle, el historial completo de la conversación se imprime.    
print("--- Message History ---")
print(chat_history)