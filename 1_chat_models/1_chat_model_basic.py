import os
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

result = model.invoke("Cuánto es 81 divido 9?")

print("Full result: ")
print(result)
print("Content only: ")
print(result.content)