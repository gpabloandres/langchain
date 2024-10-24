from langchain.prompts.chat import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

"""
PART 1: Create a ChatPromptTemplate using a template string
template = "Cuentame un chiste sobre {tema}."
prompt_template = ChatPromptTemplate.from_template(template)

print("---- Prompt from Template -----")
prompt = prompt_template.invoke({"tema": "programadores"})
print(prompt)
"""

#PART 2: Prompt with Multiple Placeholders
template_multiple = """Tu eres un asistente ayudante.
Human: Dime un {adjetivo} historia sobre un {animal}.
Asistente:"""
"""
prompt_multiple = ChatPromptTemplate.from_template(template_multiple)
prompt = prompt_multiple.invoke({"adjetivo": "gracioso", "animal": "oso peresozo"})
print("\n---- Prompt with Multiple Placeholders -----\n")
print(prompt)
"""

#PART 3: Prompt with System and Human Messages (using Tuplas)
messages = [
    ("system", "Tu eres un comediante quien cuenta chistes sobre {tema}."),
    ("human", "Cuentame {cantidad_chiste} chistes."),
]
prompt_template = ChatPromptTemplate.from_messages(messages)
prompt = prompt_template.invoke({"tema": "programadores", "cantidad_chiste": 3})
print("\n---- Prompt with System and Human Messages (Tupla) -----\n")
print(prompt)