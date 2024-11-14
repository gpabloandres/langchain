import os
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain import hub
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel
from langchain_core.tools import StructuredTool
import datetime
import wikipediaapi

# Cargar variables de entorno
load_dotenv()

# Configuración de la API de Google Generative AI
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

# Esquema de argumentos vacío para `get_current_time`
class EmptyArgs(BaseModel):
    pass

# Esquema de argumentos para `search_wikipedia`
class WikipediaArgs(BaseModel):
    query: str

# Función para obtener la hora actual sin argumentos
def get_current_time():
    """Devuelve la hora actual en formato H:MM AM/PM."""
    now = datetime.datetime.now()
    return now.strftime("%I:%M %p")

# Función para buscar en Wikipedia, con un manejo de tipo adicional
def search_wikipedia(query: str):
    """Busca en Wikipedia y devuelve el resumen del primer resultado."""
    # Aseguramos que `query` sea tratado como cadena
    query = str(query) if isinstance(query, str) else query.get("title", "")

    # Configuración de Wikipedia con el user-agent adecuado
    wiki = wikipediaapi.Wikipedia('en', user_agent="MiApp/1.0 (miemail@dominio.com)")
    page = wiki.page(query)
    
    if page.exists():
        return page.summary[:100]  # Limita a 100 caracteres para brevedad
    else:
        return "No pude encontrar información sobre ese tema."

# Definición de herramientas usando `StructuredTool`
tools = [
    StructuredTool(
        name="Time",
        func=get_current_time,
        args_schema=EmptyArgs,  # Esquema vacío para la función `get_current_time`
        description="Útil para cuando necesitas saber la hora actual.",
    ),
    StructuredTool(
        name="Wikipedia",
        func=search_wikipedia,
        args_schema=WikipediaArgs,  # Esquema con campo `query` para `search_wikipedia`
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
