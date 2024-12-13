# Importa la función `load_dotenv` para cargar variables de entorno desde un archivo `.env`.
from dotenv import load_dotenv

# Importa las clases de mensajes que permiten estructurar las interacciones con el modelo de IA.
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# Importa el módulo necesario para utilizar el modelo de chat de Google generativo AI.
from langchain_google_genai import ChatGoogleGenerativeAI

# Carga las variables de entorno del archivo `.env`, lo que permite configurar
# credenciales u otros parámetros necesarios para la API.
load_dotenv()

# Crea una instancia del modelo `ChatGoogleGenerativeAI` con parámetros personalizados.
# `model="gemini-1.5-flash"` selecciona un modelo específico (Gemini 1.5).
# `temperature=0.3` establece el nivel de aleatoriedad en las respuestas generadas,
# donde valores más bajos producen resultados más consistentes.
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

# Define una lista de mensajes que representan el historial de la conversación.
# Este historial incluye mensajes del sistema (indicaciones para el modelo),
# mensajes del usuario (HumanMessage), y respuestas previas generadas por la IA (AIMessage).

"""
# Comentado: Un ejemplo inicial simple con una única pregunta y sin respuesta previa de la IA.
messages = [
    SystemMessage(content="Resuelve el siguiente problema matemático"),
    HumanMessage(content="Cuánto es 81 dividido 9?"),
]
"""

# En este caso, la lista incluye:
# 1. Una instrucción general del sistema sobre el propósito de la conversación.
# 2. Un mensaje del usuario solicitando el cálculo de "81 dividido 9".
# 3. Una respuesta generada previamente por la IA que responde "81 dividido 9 es 9".
# 4. Otro mensaje del usuario solicitando un cálculo adicional: "10 multiplicado por 5".

# Historial de mensajes que define el contexto de la conversación:
messages = [
    SystemMessage(content="Resuelve el siguiente problema matemático"),
    HumanMessage(content="Cuánto es 81 dividido 9?"),
    AIMessage(content="81 dividido 9 es 9"),
    HumanMessage(content="Cuánto es 10 multiplicado por 5?")
]

# Invoca al modelo con el historial de mensajes y genera la respuesta correspondiente.
# El modelo procesa el contexto dado y responde con un mensaje basado en los inputs.
result = model.invoke(messages)

# Imprime el contenido de la respuesta generada por la IA.
print(f"Respuesta de la IA: {result.content}")
