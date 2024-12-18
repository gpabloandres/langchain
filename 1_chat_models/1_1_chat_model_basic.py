# Importar las librerías necesarias.
import os
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# Cargar las variables de entorno.
load_dotenv()

# Configurar el modelo de IA: Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

# Realizar una query al modelo.
result = model.invoke("Cuánto es 81 divido 9?")

#print("Full result: ")
#print(result)
print("Content only: ")
print(result.content)