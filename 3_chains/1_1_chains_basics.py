import os
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "Tu eres un comediante que cuenta chistes sobre {tema}."),
        ("human", "Cuentame {numero_chiste} chistes."),
    ]
)

chain = prompt_template | model | StrOutputParser()

result = chain.invoke({"tema": "abogados", "numero_chiste": 2})

print(result)