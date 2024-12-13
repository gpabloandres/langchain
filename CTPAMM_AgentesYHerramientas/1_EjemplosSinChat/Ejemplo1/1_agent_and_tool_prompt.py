# Importa las bibliotecas necesarias.
import os  # Para interactuar con el sistema operativo, como leer variables de entorno.
import funciones  # Para acceder a las funciones que definen las herramientas.
from dotenv import load_dotenv  # Para cargar variables de entorno desde un archivo .env.
import google.generativeai as genai  # Biblioteca para trabajar con modelos generativos de Google.
from langchain_google_genai import ChatGoogleGenerativeAI  # Para integrar Google Generative AI con LangChain.
from langchain import hub  # Para acceder a prompts almacenados en el hub de LangChain.
from langchain.agents import (  # Para crear y ejecutar agentes en LangChain.
    AgentExecutor,  # Clase para ejecutar agentes con herramientas.
    create_react_agent,  # Método para crear un agente de tipo ReAct.
)
from langchain_core.tools import Tool  # Clase para definir herramientas personalizadas en LangChain.

# Carga las variables de entorno desde el archivo .env.
load_dotenv()  

# Configura la API de Google Generative AI con una clave almacenada en una variable de entorno.
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Configura el modelo de lenguaje utilizando la biblioteca ChatGoogleGenerativeAI.
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

# Define una lista de herramientas para el agente.
tools = [
    Tool(
        name="Time",
        func=funciones.get_current_time,
        description="Useful for when you need to know the current time",
    ),
]

# Descarga un prompt del hub de LangChain.
prompt = hub.pull("hwchase17/react")  # Obtiene el prompt de tipo ReAct creado por hwchase17.

# Crea un agente ReAct configurado con el LLM, las herramientas y el prompt descargado.
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
    stop_sequence=True,
)

# Crea un ejecutor para el agente con las herramientas configuradas.
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
)

# Invoca al agente con una entrada específica para obtener la hora actual.
response = agent_executor.invoke({"input": "What time is it?"})

# Imprime la respuesta del agente.
print("response:", response)
