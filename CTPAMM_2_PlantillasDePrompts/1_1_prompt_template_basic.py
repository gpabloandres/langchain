from langchain.prompts.chat import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

#PART 1: Create a ChatPromptTemplate using a template string

# Se define una cadena de texto con un marcador de posición ({tema}), que permite personalizar dinámicamente el texto reemplazando {tema} con un valor específico.
template = "Cuentame un chiste sobre {tema}."

# Este método toma la plantilla (template) definida anteriormente y la convierte en un objeto ChatPromptTemplate.
prompt_template = ChatPromptTemplate.from_template(template)

print("---- Prompt from Template -----")

# Este método toma un diccionario con los valores necesarios para reemplazar los marcadores de posición en la plantilla.
prompt = prompt_template.invoke({"tema": "programadores"})

# Se imprime el mensaje dinámico generado tras sustituir el marcador en la plantilla con el valor proporcionado.
print(prompt)
