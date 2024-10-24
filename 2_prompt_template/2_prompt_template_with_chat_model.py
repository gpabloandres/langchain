import os
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts.chat import ChatPromptTemplate

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

"""
PART 1: Create a ChatPromptTemplate using a template string
template = "Cuentame un chiste sobre {tema}."
prompt_template = ChatPromptTemplate.from_template(template)

print("---- Prompt from Template -----")
prompt = prompt_template.invoke({"tema": "hermanos menores"})
result = model.invoke(prompt)
print(result.content)
"""

#PART 2: Prompt with Multiple Placeholders
#template_multiple = """Tu eres un asistente ayudante.
#Human: Dime un {adjetivo} historia sobre un {animal}.
#Asistente:"""
"""prompt_multiple = ChatPromptTemplate.from_template(template_multiple)
prompt = prompt_multiple.invoke({"adjetivo": "gracioso", "animal": "oso peresozo"})
print("\n---- Prompt with Multiple Placeholders -----\n")
result = model.invoke(prompt)
print(result.content)
"""

#PART 3: Prompt with System and Human Messages (using Tuplas)
messages = [
    ("system", "Tu eres un comediante quien cuenta chistes sobre {tema}."),
    ("human", "Cuentame {cantidad_chiste} chistes."),
]
prompt_template = ChatPromptTemplate.from_messages(messages)
prompt = prompt_template.invoke({"tema": "abogados", "cantidad_chiste": 2})
print("\n---- Prompt with System and Human Messages (Tupla) -----\n")
result = model.invoke(prompt)
print(result.content)