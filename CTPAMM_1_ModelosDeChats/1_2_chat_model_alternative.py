from langchain_groq import ChatGroq
from dotenv import load_dotenv  # Importa load_dotenv para cargar variables de entorno.

# Carga las variables de entorno desde el archivo .env, incluyendo la clave de la API.
load_dotenv()

model = ChatGroq(model="mixtral-8x7b-32768", temperature=0)

result = model.invoke("Hola quién sos?")

print(f"Respuesta desde Groq: {result.content}")