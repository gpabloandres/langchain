from langchain.prompts.chat import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

#PART 2: Prompt with System and Human Messages (using Tuplas)

# Se define una lista de tuplas que especifican dos tipos de mensajes:
# Este formato permite estructurar las interacciones de forma clara, separando el contexto general (rol) del contenido de la interacción.
messages = [
    ("system", "Tu eres un comediante quien cuenta chistes sobre {tema}."),  # Define el rol del modelo, en este caso, como "comediante".
    ("human", "Cuentame {cantidad_chiste} chistes."),  # Representa la interacción del usuario, solicitando que se cuente una cantidad específica de chistes.
]

# Convierte la lista de tuplas de mensajes en un objeto
prompt_template = ChatPromptTemplate.from_messages(messages)

# Se utiliza para sustituir los marcadores de posición ({tema} y {cantidad_chiste}) con valores específicos proporcionados como un diccionario.
"""
El resultado será una estructura lista para enviar al modelo de lenguaje que incluye:
Un mensaje de sistema que establece el contexto: "Tú eres un comediante quien cuenta chistes sobre programadores."
Un mensaje humano que indica la acción deseada: "Cuentame 3 chistes."
"""
prompt = prompt_template.invoke({"tema": "programadores", "cantidad_chiste": 3})

print("\n---- Prompt with System and Human Messages (Tupla) -----\n")

print(prompt)
