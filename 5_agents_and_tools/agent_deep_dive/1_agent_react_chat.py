import os
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain import hub
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import Tool
import datetime
import wikipediaapi

# Cargar variables de entorno
load_dotenv()

# Configuración de la API de Google Generative AI
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

# Función auxiliar para obtener la hora actual
def get_current_time(input=None):
    """Devuelve la hora actual en formato H:MM AM/PM."""
    now = datetime.datetime.now()
    return now.strftime("%I:%M %p")

# Función auxiliar para buscar en Wikipedia
def search_wikipedia(query):
    """Busca en Wikipedia y devuelve el resumen del primer resultado."""
    wiki = wikipediaapi.Wikipedia('en')  # 'es' para español, cambia a 'en' para inglés
    page = wiki.page(query)
    
    if page.exists():
        return page.summary[:100]  # Limita a 500 caracteres para brevedad
    else:
        return "No pude encontrar información sobre ese tema."

# Definición de herramientas
tools = [
    Tool(
        name="Time",
        func=get_current_time,
        description="Útil para cuando necesitas saber la hora actual.",
    ),
    Tool(
        name="Wikipedia",
        func=search_wikipedia,
        description="Útil para cuando necesitas información sobre un tema.",
    )
]

# Cargar el prompt estructurado para el agente
prompt = hub.pull("hwchase17/structured-chat-agent")

# Crear y configurar el agente
agent = create_structured_chat_agent(
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

# Configuración inicial del contexto
initial_message = "Eres un asistente de IA que puede proporcionar respuestas útiles usando herramientas disponibles. " \
                  "Si no puedes responder, puedes usar las siguientes herramientas: Time y Wikipedia."
context = [SystemMessage(content=initial_message)]

# Bucle de interacción con el usuario sin memoria persistente
while True:
    user_input = input("Usuario: ")
    if user_input.lower() == "exit":
        print("Terminando la conversación.")
        break

    # Agregar el mensaje del usuario al contexto
    context.append(HumanMessage(content=user_input))

    # Crear un prompt de contexto unificado para el agente
    prompt_with_context = "\n".join([msg.content for msg in context])

    # Obtener la respuesta del agente
    response = agent_executor.invoke({"input": prompt_with_context})

    # Mostrar y almacenar respuesta del agente en el contexto
    bot_response = response["output"]
    print("Bot:", bot_response)
    context.append(AIMessage(content=bot_response))
